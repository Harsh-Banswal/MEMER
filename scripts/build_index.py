"""
Parallel FAISS index builder.

For 100 memes:   single-threaded is fine (~30 seconds)
For 50,000 memes: single-threaded takes 4+ hours
                  parallel takes ~20 minutes (8 cores)

How parallelism works here:
  concurrent.futures.ProcessPoolExecutor splits the meme list
  into chunks and processes each chunk in a separate CPU core.
  Each worker runs MediaPipe independently.
  Results are collected and merged into one FAISS index.

Why ProcessPoolExecutor not ThreadPoolExecutor?
  MediaPipe releases the GIL (Global Interpreter Lock) during
  inference, but Python's GIL still limits true CPU parallelism
  for threads. Processes have separate memory spaces — true
  parallelism. For CPU-bound ML inference, processes win.
"""

import json
import numpy as np
import faiss
import cv2
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.pose.estimator  import PoseEstimator
from app.pose.normalizer import normalize_pose

MEME_DIR   = Path("data/memes")
META_FILE  = Path("data/metadata.json")
INDEX_FILE = Path("data/index.faiss")
IDMAP_FILE = Path("data/id_map.json")
VECTOR_DIM = 106


def process_chunk(chunk: list[tuple[int, dict]]) -> list[tuple[int, np.ndarray]]:
    """
    Processes a chunk of memes in a worker process.

    Each worker creates its OWN PoseEstimator because MediaPipe
    objects cannot be pickled and shared across processes.

    Parameters:
      chunk: list of (metadata_index, metadata_dict) tuples

    Returns:
      list of (metadata_index, vector) for successful poses only
    """
    estimator = PoseEstimator()
    results   = []

    for meta_idx, meme in chunk:
        path = MEME_DIR / meme["filename"]
        if not path.exists():
            continue
        try:
            img_bytes = path.read_bytes()
            result    = estimator.process_bytes(img_bytes)
            vector    = normalize_pose(result) if result else None
            if vector is not None:
                results.append((meta_idx, vector))
        except Exception:
            pass  # skip corrupt images silently

    estimator.close()
    return results


def build_index(num_workers: int = None):
    """
    Builds FAISS index from all memes in data/memes/.

    num_workers: CPU cores to use. Defaults to half your cores
                 (leaving room for the OS and other tasks).
    """
    if not META_FILE.exists():
        raise FileNotFoundError("Run scripts/download_memes.py first")

    with open(META_FILE) as f:
        metadata = json.load(f)

    print(f"Building index for {len(metadata)} memes...")

    # Default to a safe limit to prevent high RAM OOM crashes on multi-core CPUs
    if num_workers is None:
        num_workers = min(3, max(1, multiprocessing.cpu_count() // 2))
    print(f"Using {num_workers} worker processes")

    # Split metadata into chunks — one per worker
    indexed = list(enumerate(metadata))
    chunk_size = max(1, len(indexed) // num_workers)
    chunks = [
        indexed[i:i + chunk_size]
        for i in range(0, len(indexed), chunk_size)
    ]

    all_results = []  # (meta_idx, vector) pairs

    # Run chunks in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_chunk, chunk): i for i, chunk in enumerate(chunks)}
        for future in as_completed(futures):
            chunk_results = future.result()
            all_results.extend(chunk_results)
            print(f"  Progress: {len(all_results)}/{len(metadata)} poses extracted", end="\r")

    print(f"\nPoses extracted: {len(all_results)} / {len(metadata)}")

    if not all_results:
        raise RuntimeError("No poses found in any image.")

    # Sort by original metadata index so id_map is consistent
    all_results.sort(key=lambda x: x[0])

    id_map  = [r[0]  for r in all_results]
    vectors = [r[1]  for r in all_results]

    # Stack → normalise → build index
    mat = np.vstack(vectors).astype(np.float32)
    faiss.normalize_L2(mat)  # in-place, required for cosine similarity

    index = faiss.IndexFlatIP(VECTOR_DIM)
    index.add(mat)

    # Save to disk
    faiss.write_index(index, str(INDEX_FILE))
    with open(IDMAP_FILE, "w") as f:
        json.dump(id_map, f)

    print(f"Index saved: {INDEX_FILE} ({index.ntotal} vectors)")
    print("Done. You never need to run this again unless you add new memes.")


if __name__ == "__main__":
    build_index()