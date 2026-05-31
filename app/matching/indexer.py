"""
Builds the FAISS index from meme images.

What is FAISS?
  Facebook AI Similarity Search — a library for finding the most
  similar vectors in a large collection, extremely fast.

  Normal approach: compare your pose against every meme one by one.
  For 10,000 memes that is 10,000 dot products per frame = slow.

  FAISS approach: organise vectors into a tree/cluster structure
  during indexing so at query time you only check a small fraction
  of all vectors. 10,000 memes → ~2ms query time.

Why IndexFlatIP (inner product)?
  FAISS has many index types:
  - IndexFlatL2:  Euclidean distance (straight-line distance between vectors)
  - IndexFlatIP:  Inner product = cosine similarity when vectors are normalised
  - IndexIVFFlat: Approximate, faster for millions of vectors
  - IndexHNSW:    Graph-based, fastest approximate search

  We use IndexFlatIP because:
  1. Our vectors are L2-normalised (unit length)
  2. Inner product of two unit vectors = cosine similarity
  3. Cosine similarity is what we want — it measures angle between poses,
     not magnitude. Two people doing the same pose at different distances
     have the same angle but different magnitudes.
  4. IndexFlatIP is exact (not approximate) — fine for < 100k memes.
"""

import json
import numpy as np
import faiss
from pathlib import Path
from typing import Optional

# Import our existing tools from Phase 1
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from app.pose.estimator  import PoseEstimator
from app.pose.normalizer import normalize_pose


MEME_DIR   = Path("data/memes")
META_FILE  = Path("data/metadata.json")
INDEX_FILE = Path("data/index.faiss")
VECS_FILE  = Path("data/vectors.npy")
IDMAP_FILE = Path("data/id_map.json")

# Vector dimension — must match normalizer output
# 11 body × 2 + 21 left hand × 2 + 21 right hand × 2 = 106
VECTOR_DIM = 106


def build_index():
    """
    Main function. Steps:
    1. Load metadata (list of meme filenames)
    2. For each meme image, run PoseEstimator + normalize_pose
    3. Collect all valid vectors
    4. L2-normalise them (required for cosine similarity via IndexFlatIP)
    5. Build FAISS index and save to disk
    6. Save id_map so we can go from FAISS result index → meme filename
    """

    # Load metadata
    if not META_FILE.exists():
        raise FileNotFoundError(f"Run scripts/download_memes.py first — {META_FILE} not found")

    with open(META_FILE) as f:
        metadata = json.load(f)

    print(f"Processing {len(metadata)} memes...")

    estimator = PoseEstimator()
    vectors   = []   # list of np.float32 arrays, each shape (106,)
    id_map    = []   # maps FAISS index position → metadata index

    for i, meme in enumerate(metadata):
        path = MEME_DIR / meme["filename"]
        if not path.exists():
            print(f"  [{i+1}] MISSING file: {path.name} — skip")
            continue

        # Read image as raw bytes — same format as webcam frames
        # This lets us reuse process_bytes() without modification
        img_bytes = path.read_bytes()

        result = estimator.process_bytes(img_bytes)
        vector = normalize_pose(result) if result else None

        if vector is None:
            # No human pose detected in this meme image
            # Common for memes with text only, animals, or abstract images
            print(f"  [{i+1}] No pose: {meme['name']}")
            continue

        vectors.append(vector)
        id_map.append(i)   # remember which metadata entry this vector belongs to
        print(f"  [{i+1}] OK: {meme['name']}")

    estimator.close()

    if len(vectors) == 0:
        raise RuntimeError("No poses detected in any meme. Check your meme images.")

    print(f"\nPoses extracted: {len(vectors)} / {len(metadata)}")

    # Stack into a 2D matrix: shape (num_memes, 106)
    # FAISS requires a contiguous float32 matrix
    mat = np.vstack(vectors).astype(np.float32)

    # L2-normalise every row so each vector has unit length
    # After this, inner product == cosine similarity
    # faiss.normalize_L2 modifies mat in-place
    faiss.normalize_L2(mat)

    # Build FAISS flat inner-product index
    # IndexFlatIP = brute-force but with BLAS-accelerated matrix multiply
    # For < 100k vectors this is fast enough (< 5ms per query)
    index = faiss.IndexFlatIP(VECTOR_DIM)

    # Add all vectors to the index
    # FAISS assigns sequential IDs: 0, 1, 2, ...
    index.add(mat)

    print(f"FAISS index built: {index.ntotal} vectors, dim={VECTOR_DIM}")

    # Save everything to disk
    faiss.write_index(index, str(INDEX_FILE))
    np.save(str(VECS_FILE), mat)
    with open(IDMAP_FILE, "w") as f:
        json.dump(id_map, f)

    print(f"Saved: {INDEX_FILE}, {VECS_FILE}, {IDMAP_FILE}")
    print("Index build complete. Ready for Phase 3.")


if __name__ == "__main__":
    build_index()