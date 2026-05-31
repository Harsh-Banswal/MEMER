import pytest
import numpy as np
from pathlib import Path
import sys

# Ensure project root is in the path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.pose.estimator import PoseEstimator, PoseResult


def test_estimator_initialization():
    """
    Test that the PoseEstimator can be successfully initialized and closed.
    """
    estimator = PoseEstimator()
    assert estimator.holistic is not None
    estimator.close()


def test_estimator_invalid_bytes():
    """
    Test that processing corrupt or invalid bytes returns None gracefully.
    """
    estimator = PoseEstimator()
    corrupt_bytes = b"not-a-valid-image-format-bytes-xyz-123"
    
    result = estimator.process_bytes(corrupt_bytes)
    assert result is None
    estimator.close()
