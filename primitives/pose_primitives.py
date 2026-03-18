import numpy as np
from dataclasses import dataclass
from typing import Optional
from perception.frame_data import PoseData, PoseLM


@dataclass
class PosePrimitives:
    arms_spread_angle:  Optional[float]
    left_wrist_height:  Optional[float]
    right_wrist_height: Optional[float]
    arms_raised:        bool
    stance:             str
    wrists_distance:    Optional[float]


class PosePrimitivesExtractor:

    ANGLE_WIDE_MIN   = 140
    WRIST_RAISED_MIN = 0.05
    WRIST_LOW_MAX    = -0.05
    WRISTS_CLOSE_MAX = 0.4

    def extract(self, pose: Optional[PoseData]) -> PosePrimitives:

        if pose is None or not pose.upper_body_visible:
            return PosePrimitives(None, None, None, False, "occluded", None)

        lm = pose.landmarks

        l_sh = lm[PoseLM.LEFT_SHOULDER,  :3]
        r_sh = lm[PoseLM.RIGHT_SHOULDER, :3]
        l_el = lm[PoseLM.LEFT_ELBOW,     :3]
        r_el = lm[PoseLM.RIGHT_ELBOW,    :3]
        l_wr = lm[PoseLM.LEFT_WRIST,     :3]
        r_wr = lm[PoseLM.RIGHT_WRIST,    :3]

        mid = (l_sh + r_sh) / 2.0

        # ── Angle bras ─────────────────────────────
        v_l = l_el - mid
        v_r = r_el - mid

        mag = np.linalg.norm(v_l) * np.linalg.norm(v_r)
        if mag > 1e-6:
            arms_angle = float(
                np.degrees(
                    np.arccos(np.clip(np.dot(v_l, v_r) / mag, -1, 1))
                )
            )
        else:
            arms_angle = None

        # ── Hauteur poignets ──────────────────────
        sy  = float(mid[1])
        l_h = float(sy - l_wr[1])
        r_h = float(sy - r_wr[1])

        arms_raised = (
            l_h > self.WRIST_RAISED_MIN
            and r_h > self.WRIST_RAISED_MIN
        )

        # ── Distance poignets normalisée ──────────
        sw = float(np.linalg.norm(r_sh[:2] - l_sh[:2]))
        wd = (
            float(np.linalg.norm(r_wr[:2] - l_wr[:2]) / sw)
            if sw > 1e-6 else None
        )

        stance = self._stance(arms_angle, l_h, r_h, wd)

        return PosePrimitives(
            arms_angle,
            l_h,
            r_h,
            arms_raised,
            stance,
            wd,
        )


    def _stance(self, angle, lh, rh, wd):

        if angle is None:
            return "neutral"

        if lh > self.WRIST_RAISED_MIN and rh > self.WRIST_RAISED_MIN:
            if wd is None or wd >= self.WRISTS_CLOSE_MAX:
                return "arms_raised"

        if wd is not None and wd < 0.10 and abs(lh) < 0.1:  
            return "arms_cross"

        if wd is not None and wd < self.WRISTS_CLOSE_MAX:
            if lh < self.WRIST_LOW_MAX:
                return "charge"
            return "arms_together"

        if angle > self.ANGLE_WIDE_MIN:
            return "arms_wide"

        return "neutral"