import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict

PROJECT_ROOT = Path(__file__).parent.parent.parent
INDEX_FILE   = PROJECT_ROOT / "data/index.faiss"
META_FILE    = PROJECT_ROOT / "data/metadata.json"
IDMAP_FILE   = PROJECT_ROOT / "data/id_map.json"


class PoseSearcher:
    """
    Loads the built FAISS index and maps real-time pose queries
    to the closest matching memes in the dataset.
    """
    def __init__(self):
        self.index = None
        self.metadata = []
        self.id_map = []
        self.load_index()

    def load_index(self):
        if not INDEX_FILE.exists() or not META_FILE.exists() or not IDMAP_FILE.exists():
            print(f"[Searcher] Index files not found. Ensure you run scripts/build_index.py first.")
            return

        try:
            self.index = faiss.read_index(str(INDEX_FILE))
            with open(META_FILE, "r") as f:
                self.metadata = json.load(f)
            with open(IDMAP_FILE, "r") as f:
                self.id_map = json.load(f)
            print(f"[Searcher] Successfully loaded FAISS index with {self.index.ntotal} vectors.")
        except Exception as e:
            print(f"[Searcher] Error loading index files: {e}")

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Takes a normalized query vector (106,) and returns the top K matching memes.
        
        Parameters:
            query_vector: np.ndarray shape (106,) representing the user's pose
            k: number of matches to retrieve
            
        Returns:
            list of dicts containing meme metadata and similarity scores
        """
        if self.index is None:
            self.load_index()
            if self.index is None:
                return []

        # FAISS search requires shape (1, 106) and float32 type
        query = query_vector.reshape(1, -1).astype(np.float32)
        
        # Ensure L2 normalized for cosine similarity (IndexFlatIP)
        faiss.normalize_L2(query)

        # Search index
        # D: similarities (inner product of L2 normalized = cosine similarity), I: indices
        D, I = self.index.search(query, k)

        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            
            # Map index back to original metadata
            meta_idx = self.id_map[idx]
            meme = self.metadata[meta_idx].copy()
            meme["similarity"] = float(dist)
            results.append(meme)

        return results
