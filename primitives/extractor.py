import numpy as np
from dataclasses import dataclass
from typing import Optional
from collections import deque

from perception.frame_data import FrameData
from primitives.hand_primitives import HandPrimitives, HandPrimitivesExtractor
from primitives.pose_primitives import PosePrimitives, PosePrimitivesExtractor


# ── Dataclass ────────────────────────────────────────────────────────────────

@dataclass
class FullPrimitives:

    timestamp:   float
    frame_index: int

    left:  Optional[HandPrimitives]
    right: Optional[HandPrimitives]

    pose: PosePrimitives

    motion_energy: float
    bilateral_symmetry: Optional[float]

    # 🔥 NEW
    hands_close: bool
    charge_confidence: float

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def any_hand(self) -> bool:
        return self.left is not None or self.right is not None

    @property
    def both_hands(self) -> bool:
        return self.left is not None and self.right is not None

    @property
    def dominant_hand(self) -> Optional[HandPrimitives]:
        return self.right or self.left

    @property
    def is_moving(self) -> bool:
        return self.motion_energy > 0.05

    @property
    def is_charging(self) -> bool:
        return bool(self.charge_confidence > 0.6)


# ── Aggregator ───────────────────────────────────────────────────────────────

class PrimitivesAggregator:

    MAX_SPEED_REF = 2000.0
    HANDS_CLOSE_THRESHOLD = 80.0   # pixels

    def __init__(self, history_len: int = 15):
        self._hand_ex  = HandPrimitivesExtractor()
        self._pose_ex  = PosePrimitivesExtractor()
        self._history: deque[FrameData] = deque(maxlen=history_len)

    # ── Main ────────────────────────────────────────────────────────────────

    def process(self, frame_data: FrameData) -> FullPrimitives:

        prev = self._history[-1] if self._history else None
        dt   = self._delta_t(frame_data, prev)

        self._history.append(frame_data)

        left_p  = self._extract_hand(frame_data.left_hand,
                                    prev.left_hand if prev else None, dt)

        right_p = self._extract_hand(frame_data.right_hand,
                                    prev.right_hand if prev else None, dt)

        pose_p  = self._pose_ex.extract(frame_data.pose)

        motion_energy = self._motion_energy(left_p, right_p)
        symmetry      = self._bilateral_symmetry(left_p, right_p)

        # 🔥 NEW
        hands_close = self._hands_close(frame_data)
        charge_conf = self._charge_confidence(
            hands_close,
            motion_energy,
            pose_p,
            left_p,
            right_p
            )

        return FullPrimitives(
            timestamp          = frame_data.timestamp,
            frame_index        = frame_data.frame_index,
            left               = left_p,
            right              = right_p,
            pose               = pose_p,
            motion_energy      = motion_energy,
            bilateral_symmetry = symmetry,
            hands_close        = hands_close,
            charge_confidence  = charge_conf,
        )

    def reset(self):
        self._history.clear()

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _extract_hand(self, hand, prev_hand, dt):
        if hand is None:
            return None
        return self._hand_ex.extract(hand, prev_hand, dt)

    @staticmethod
    def _delta_t(current: FrameData, prev: Optional[FrameData]) -> float:
        if prev is None:
            return 0.016
        return max(current.timestamp - prev.timestamp, 1e-6)

    def _motion_energy(self, left, right) -> float:
        total  = (left.wrist_speed  if left  else 0.0)
        total += (right.wrist_speed if right else 0.0)
        return float(min(total / self.MAX_SPEED_REF, 1.0))

    def _bilateral_symmetry(self, left, right) -> Optional[float]:
        if left is None or right is None:
            return None

        l_f = np.array(left.fingers_extended,  dtype=np.float32)
        r_f = np.array(right.fingers_extended, dtype=np.float32)

        return float(np.mean(l_f == r_f[::-1]))

    # 🔥 NEW ────────────────────────────────────────────────────────────────

    def _hands_close(self, frame_data: FrameData) -> bool:

        l = frame_data.left_hand
        r = frame_data.right_hand

    # ✔️ vrai cas : 2 mains détectées
        if l is not None and r is not None:
            d = np.linalg.norm(
                l.landmarks_px[0] - r.landmarks_px[0]
            )
            return d < self.HANDS_CLOSE_THRESHOLD

    # ❌ une seule main → on ne sait pas
        return False

    def _charge_confidence(
    self,
    hands_close: bool,
    motion_energy: float,
    pose: PosePrimitives,
    left: Optional[HandPrimitives],
    right: Optional[HandPrimitives],
    ) -> float:

        score = 0.0

    # ✔️ cas 1 : vraies mains proches
        if hands_close:
            score += 0.5

    # ✔️ cas 2 : une seule main MAIS immobile → probable fusion
        if (left is None) ^ (right is None):
            if motion_energy < 0.15:
                score += 0.4

    # ✔️ pose
        if pose.stance == "charge":
            score += 0.4

        return float(min(score, 1.0))