import pytest
import numpy as np
from pathlib import Path
import sys

# Ensure project root is in the path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.matching.searcher import PoseSearcher


def test_pose_searcher_initialization():
    """
    Test that the PoseSearcher can be successfully initialized.
    """
    searcher = PoseSearcher()
    # The index might not exist yet if the indexer has not been run,
    # which is an expected state, but the class should still instantiate safely.
    assert searcher is not None


def test_search_dummy_vector():
    """
    Test that searching a dummy vector runs without crashing, and either
    returns results (if index exists) or returns empty list gracefully.
    """
    searcher = PoseSearcher()
    dummy_vector = np.random.rand(106).astype(np.float32)
    
    try:
        results = searcher.search(dummy_vector, k=3)
        assert isinstance(results, list)
        if len(results) > 0:
            assert "name" in results[0]
            assert "similarity" in results[0]
            assert "url" in results[0]
    except Exception as e:
        pytest.fail(f"Search raised an unexpected exception: {e}")
