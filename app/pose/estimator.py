import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import Optional

mp_holistic = mp.solutions.holistic

@dataclass
class PoseResult:
    body_landmarks: np.ndarray     # shape (33, 2) — body keypoints
    left_hand: np.ndarray          # shape (21, 2) — left hand, zeros if not detected
    right_hand: np.ndarray         # shape (21, 2) — right hand, zeros if not detected
    visibility: np.ndarray         # shape (33,)   — body landmark confidence
    left_hand_detected: bool
    right_hand_detected: bool
    valid: bool


class PoseEstimator:
    def __init__(self, min_detection_confidence=0.4, min_tracking_confidence=0.4):
        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_bytes(self, frame_bytes: bytes) -> Optional[PoseResult]:
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return None

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        results = self.holistic.process(frame_rgb)

        # Body landmarks
        if not results.pose_landmarks:
            return PoseResult(
                body_landmarks=np.zeros((33, 2)),
                left_hand=np.zeros((21, 2)),
                right_hand=np.zeros((21, 2)),
                visibility=np.zeros(33),
                left_hand_detected=False,
                right_hand_detected=False,
                valid=False
            )

        body_lm = np.array(
            [[lm.x * w, lm.y * h] for lm in results.pose_landmarks.landmark],
            dtype=np.float32
        )
        visibility = np.array(
            [lm.visibility for lm in results.pose_landmarks.landmark],
            dtype=np.float32
        )

        # Left hand
        if results.left_hand_landmarks:
            left_hand = np.array(
                [[lm.x * w, lm.y * h] for lm in results.left_hand_landmarks.landmark],
                dtype=np.float32
            )
            left_detected = True
        else:
            left_hand = np.zeros((21, 2), dtype=np.float32)
            left_detected = False

        # Right hand
        if results.right_hand_landmarks:
            right_hand = np.array(
                [[lm.x * w, lm.y * h] for lm in results.right_hand_landmarks.landmark],
                dtype=np.float32
            )
            right_detected = True
        else:
            right_hand = np.zeros((21, 2), dtype=np.float32)
            right_detected = False

        return PoseResult(
            body_landmarks=body_lm,
            left_hand=left_hand,
            right_hand=right_hand,
            visibility=visibility,
            left_hand_detected=left_detected,
            right_hand_detected=right_detected,
            valid=True
        )

    def close(self):
        self.holistic.close()