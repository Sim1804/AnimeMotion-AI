from dataclasses import dataclass
from typing import Optional
import numpy as np

class HandLM:
    WRIST        = 0
    THUMB_CMC    = 1;  THUMB_MCP   = 2;  THUMB_IP    = 3;  THUMB_TIP   = 4
    INDEX_MCP    = 5;  INDEX_PIP   = 6;  INDEX_DIP   = 7;  INDEX_TIP   = 8
    MIDDLE_MCP   = 9;  MIDDLE_PIP  = 10; MIDDLE_DIP  = 11; MIDDLE_TIP  = 12
    RING_MCP     = 13; RING_PIP    = 14; RING_DIP    = 15; RING_TIP    = 16
    PINKY_MCP    = 17; PINKY_PIP   = 18; PINKY_DIP   = 19; PINKY_TIP   = 20

class PoseLM:
    NOSE            = 0
    LEFT_SHOULDER   = 11; RIGHT_SHOULDER  = 12
    LEFT_ELBOW      = 13; RIGHT_ELBOW     = 14
    LEFT_WRIST      = 15; RIGHT_WRIST     = 16
    LEFT_HIP        = 23; RIGHT_HIP       = 24
    LEFT_KNEE       = 25; RIGHT_KNEE      = 26

@dataclass
class HandData:
    landmarks: np.ndarray
    landmarks_px: np.ndarray
    handedness: str
    score: float

@dataclass
class PoseData:
    landmarks: np.ndarray
    landmarks_px: np.ndarray
    upper_body_visible: bool

@dataclass
class FrameData:
    timestamp: float
    frame_index: int
    frame_rgb: np.ndarray
    frame_h: int
    frame_w: int
    left_hand: Optional[HandData] = None
    right_hand: Optional[HandData] = None
    pose: Optional[PoseData] = None

@property
def both_hands_visible(self) -> bool:
    return self.left_hand is not None and self.right_hand is not None

@property
def any_hand_visible(self) -> bool:
    return self.left_hand is not None or self.right_hand is not None