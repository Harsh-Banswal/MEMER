import numpy as np
from .estimator import PoseResult

LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12

UPPER_BODY_INDICES = [0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 16]

def normalize_pose(result: PoseResult, visibility_threshold=0.4):
    if not result.valid:
        return None

    lm  = result.body_landmarks  # (33, 2)
    vis = result.visibility       # (33,)

    if vis[LEFT_SHOULDER] < visibility_threshold or vis[RIGHT_SHOULDER] < visibility_threshold:
        return None

    # Origin and scale from shoulders
    origin = (lm[LEFT_SHOULDER] + lm[RIGHT_SHOULDER]) / 2.0
    shoulder_width = np.linalg.norm(lm[LEFT_SHOULDER] - lm[RIGHT_SHOULDER])
    if shoulder_width < 1e-6:
        return None

    # Normalize body landmarks
    upper_lm  = lm[UPPER_BODY_INDICES]
    upper_vis = vis[UPPER_BODY_INDICES]
    body_norm = (upper_lm - origin) / shoulder_width
    body_weighted = body_norm * upper_vis[:, np.newaxis]  # (11, 2)

    # Normalize hand landmarks using the same origin + scale only if they are detected
    if getattr(result, "left_hand_detected", False):
        left_norm = (result.left_hand - origin) / shoulder_width
    else:
        left_norm = np.zeros((21, 2), dtype=np.float32)

    if getattr(result, "right_hand_detected", False):
        right_norm = (result.right_hand - origin) / shoulder_width
    else:
        right_norm = np.zeros((21, 2), dtype=np.float32)

    # Concatenate everything into one vector
    vector = np.concatenate([
        body_weighted.flatten(),   # 22 values  (11 landmarks × 2)
        left_norm.flatten(),       # 42 values  (21 landmarks × 2)
        right_norm.flatten(),      # 42 values  (21 landmarks × 2)
    ])                             # total: 106 values

    return vector.astype(np.float32)