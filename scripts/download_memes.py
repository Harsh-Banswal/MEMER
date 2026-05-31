"""
Two-mode meme collector:
  Mode A — Imgflip API (100 memes, instant, online)
  Mode B — Local folder scan (thousands of memes from Kaggle ZIP)

Why separate modes?
  Imgflip gives clean named templates but only 100.
  Kaggle gives thousands but filenames are meaningless (just IDs).
  We handle both and merge into one metadata.json.
"""

import os
import json
import time
import shutil
import urllib.request
import zipfile
from pathlib import Path

MEME_DIR  = Path("data/memes")
META_FILE = Path("data/metadata.json")
MEME_DIR.mkdir(parents=True, exist_ok=True)

# Supported image formats — we skip GIFs for now (Phase 5 adds them back)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


# ─── Mode A: Imgflip ───────────────────────────────────────────────
def collect_from_imgflip() -> list[dict]:
    """
    Fetches 100 meme templates from Imgflip's free public API.
    Returns list of metadata dicts.

    Why keep this even with Kaggle?
    Imgflip memes are the most recognisable — Distracted Boyfriend,
    Drake, This Is Fine. They are the gold standard test cases.
    """
    print("\n[Imgflip] Fetching 100 templates...")
    url = "https://api.imgflip.com/get_memes"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        memes = data["data"]["memes"]
    except Exception as e:
        print(f"  Imgflip failed: {e}")
        return []

    metadata = []
    for i, meme in enumerate(memes):
        ext      = Path(meme["url"]).suffix or ".jpg"
        filename = f"imgflip_{meme['id']}{ext}"
        dest     = MEME_DIR / filename

        if not dest.exists():
            try:
                urllib.request.urlretrieve(meme["url"], dest)
                time.sleep(0.05)
            except Exception as e:
                print(f"  [{i+1}] Failed: {meme['name']} — {e}")
                continue

        metadata.append({
            "filename": filename,
            "name":     meme["name"],
            "source":   "imgflip",
            "width":    meme["width"],
            "height":   meme["height"],
        })
        print(f"  [{i+1}/100] {meme['name']}")

    print(f"[Imgflip] Done: {len(metadata)} memes")
    return metadata


# ─── Mode B: Local Kaggle folder ──────────────────────────────────
def collect_from_local(source_dir: str) -> list[dict]:
    """
    Scans a local folder for meme images (from Kaggle ZIP extract).
    Copies valid images into data/memes/ and builds metadata.

    Why copy instead of reference in-place?
    Keeping all memes in one folder (data/memes/) makes the FAISS
    index portable — you can move the project and everything works.
    If the Kaggle folder is large (10GB+), set COPY=False below
    and just reference the original paths instead.

    source_dir: path to your extracted Kaggle folder
                e.g. "C:/Downloads/reddit-memes-dataset"
    """
    COPY = True   # set False if dataset is huge and disk space is limited

    source = Path(source_dir)
    if not source.exists():
        print(f"[Local] Folder or ZIP not found: {source_dir}")
        return []

    zip_prefix = "local"
    # If the source is a ZIP file, extract it first
    if source.is_file() and source.suffix.lower() == ".zip":
        zip_prefix = source.stem.replace(" ", "_").replace("(", "").replace(")", "")
        extract_dir = MEME_DIR.parent / "extracted_memes" / zip_prefix
        print(f"[Local] Extracting ZIP: {source} to {extract_dir}...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(source, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"[Local] Extraction complete!")
            source = extract_dir
        except Exception as e:
            print(f"[Local] Failed to extract ZIP: {e}")
            return []

    # Recursively find all image files
    all_images = [
        p for p in source.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS
    ]
    print(f"\n[Local] Found {len(all_images)} images in {source_dir}")

    metadata = []
    for i, img_path in enumerate(all_images):
        # Incorporate parent directory and zip name to prevent collisions between different dataset ZIPs
        parent_name = img_path.parent.name
        raw_name = f"local_{zip_prefix}_{parent_name}_{img_path.stem}"
        
        # Sanitise filename — remove spaces and special chars
        filename = "".join(
            c if c.isalnum() or c in "._-" else "_"
            for c in raw_name
        )
        filename = f"{filename}{img_path.suffix.lower()}"
        dest = MEME_DIR / filename

        if not dest.exists():
            if COPY:
                try:
                    shutil.copy2(img_path, dest)
                except Exception as e:
                    print(f"  [{i+1}] Copy failed: {e}")
                    continue
            else:
                # Just reference original path — don't copy
                dest = img_path
                filename = str(img_path)

        # Use folder name as meme "name" — often descriptive in Kaggle datasets
        name = img_path.parent.name or img_path.stem
        name = name.replace("_", " ").replace("-", " ").title()

        metadata.append({
            "filename": filename if COPY else str(img_path),
            "name":     name,
            "source":   "local",
        })

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(all_images)}...")

    print(f"[Local] Done: {len(metadata)} memes")
    return metadata


# ─── Main ─────────────────────────────────────────────────────────
def main():
    import sys

    # Check if user passed a local folder path
    # Usage: python scripts/download_memes.py /path/to/kaggle/folder
    local_dir = sys.argv[1] if len(sys.argv) > 1 else None

    # Load existing metadata if it exists to allow merging multiple datasets
    metadata = []
    if META_FILE.exists():
        try:
            with open(META_FILE, "r") as f:
                metadata = json.load(f)
            print(f"[Merge] Loaded {len(metadata)} existing memes from {META_FILE}")
        except Exception as e:
            print(f"[Merge] Failed to load existing metadata: {e}")
            metadata = []

    # Always ensure Imgflip templates are loaded
    has_imgflip = any(m.get("source") == "imgflip" for m in metadata)
    if not has_imgflip:
        print("[Imgflip] Loading default templates...")
        metadata = collect_from_imgflip() + metadata

    # If local folder given, scan and append it
    if local_dir:
        metadata += collect_from_local(local_dir)
    else:
        print("\n[Info] No local folder given.")
        print("       To add Kaggle dataset:")
        print("       python scripts/download_memes.py /path/to/your/kaggle/folder")

    # Remove duplicates by filename
    seen = set()
    unique = []
    for m in metadata:
        if m["filename"] not in seen:
            seen.add(m["filename"])
            unique.append(m)

    # Save merged metadata
    with open(META_FILE, "w") as f:
        json.dump(unique, f, indent=2)

    print(f"\n{'='*40}")
    print(f"Total memes collected: {len(unique)}")
    print(f"Saved to: {META_FILE}")
    print(f"\nNext step: python scripts/build_index.py")


if __name__ == "__main__":
    main()