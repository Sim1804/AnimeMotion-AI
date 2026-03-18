import numpy as np
from dataclasses import dataclass
from typing import Optional
from perception.frame_data import PoseData, PoseLM


@dataclass
class PosePrimitives:
    arms_spread_angle: Optional[float]
    left_wrist_height: Optional[float]
    right_wrist_height: Optional[float]
    arms_raised: bool
    stance: str
    wrists_distance: Optional[float]
    symmetry: float
    stability: float
    confidence: float


class PosePrimitivesExtractor:

    ANGLE_CROSS_MAX  = 30
    ANGLE_WIDE_MIN   = 140
    WRIST_RAISED_MIN = 0.05
    WRIST_LOW_MAX    = -0.05
    WRISTS_CLOSE_MAX = 0.4

    def __init__(self):
        self.prev_angle = None

    def extract(self, pose: Optional[PoseData]) -> PosePrimitives:

        if pose is None or not pose.upper_body_visible:
            return PosePrimitives(None, None, None, False, "occluded", None, 0, 0, 0)

        lm = pose.landmarks

        l_sh = lm[PoseLM.LEFT_SHOULDER, :3]
        r_sh = lm[PoseLM.RIGHT_SHOULDER, :3]
        l_el = lm[PoseLM.LEFT_ELBOW, :3]
        r_el = lm[PoseLM.RIGHT_ELBOW, :3]
        l_wr = lm[PoseLM.LEFT_WRIST, :3]
        r_wr = lm[PoseLM.RIGHT_WRIST, :3]

        mid = (l_sh + r_sh) / 2.0

        v_l = l_el - mid
        v_r = r_el - mid

        mag = np.linalg.norm(v_l) * np.linalg.norm(v_r)
        angle = float(np.degrees(np.arccos(np.clip(np.dot(v_l, v_r) / mag, -1, 1)))) if mag > 1e-6 else None

        sy = float(mid[1])
        l_h = float(sy - l_wr[1])
        r_h = float(sy - r_wr[1])

        arms_raised = l_h > self.WRIST_RAISED_MIN and r_h > self.WRIST_RAISED_MIN

        sw = float(np.linalg.norm(r_sh[:2] - l_sh[:2]))
        wd = float(np.linalg.norm(r_wr[:2] - l_wr[:2])) / sw if sw > 1e-6 else None

        symmetry = 1.0 - abs(l_h - r_h)

        if self.prev_angle is None or angle is None:
            stability = 1.0
        else:
            stability = 1.0 - min(abs(angle - self.prev_angle) / 180.0, 1.0)

        self.prev_angle = angle

        stance = self._stance(angle, l_h, r_h, wd, l_wr, r_wr)

        confidence = np.clip(symmetry * stability, 0, 1)

        return PosePrimitives(
            angle, l_h, r_h, arms_raised, stance, wd,
            symmetry, stability, confidence
        )

    def _stance(self, angle, lh, rh, wd, l_wr, r_wr):

        if angle is None:
            return "neutral"

        if lh > self.WRIST_RAISED_MIN and rh > self.WRIST_RAISED_MIN:
            return "arms_raised"

        if angle < self.ANGLE_CROSS_MAX:
            if l_wr[0] > r_wr[0]:
                return "arms_cross"
            return "arms_together"

        if angle > self.ANGLE_WIDE_MIN:
            return "arms_wide"

        if wd is not None and wd < self.WRISTS_CLOSE_MAX:
            return "charge" if lh < self.WRIST_LOW_MAX else "hands_close"

        return "neutral"
