
import numpy as np
from dataclasses import dataclass
from typing import Optional

from perception.frame_data import HandData, HandLM


@dataclass
class HandPrimitives:
    fingers_extended: list[bool]
    fingers_count: int
    openness: float
    palm_normal: np.ndarray
    palm_facing: str
    shape_name: Optional[str]
    wrist_velocity: np.ndarray
    wrist_speed: float


class HandPrimitivesExtractor:

    PINCH_THRESHOLD = 0.06
    EXTENSION_RATIO = 1.15


    def extract(
        self,
        hand: HandData,
        prev_hand: Optional[HandData],
        dt: float,
    ) -> HandPrimitives:

        lm   = hand.landmarks      # (21, 3)
        lm_p = hand.landmarks_px   # (21, 2)

        fingers = self._fingers_extended(lm)
        fingers_count = sum(fingers)

        openness = fingers_count / 5.0
        normal   = self._palm_normal(lm)
        facing   = self._palm_facing(normal)

        shape = self._classify_shape(fingers, lm)

        velocity, speed = self._compute_velocity(lm_p, prev_hand, dt)

        return HandPrimitives(
            fingers_extended=fingers,
            fingers_count=fingers_count,
            openness=openness,
            palm_normal=normal,
            palm_facing=facing,
            shape_name=shape,
            wrist_velocity=velocity,
            wrist_speed=speed,
        )


    def _fingers_extended(self, lm: np.ndarray) -> list[bool]:
        wrist = lm[HandLM.WRIST, :2]

        def extended(tip, pip):
            tip_dist = np.linalg.norm(lm[tip, :2] - wrist)
            pip_dist = np.linalg.norm(lm[pip, :2] - wrist)
            return tip_dist > pip_dist * self.EXTENSION_RATIO

        # Pouce (cas spécial)
        thumb_open = (
            np.linalg.norm(
                lm[HandLM.THUMB_TIP, :2] - lm[HandLM.INDEX_MCP, :2]
            ) > 0.08
        )

        return [
            thumb_open,
            extended(HandLM.INDEX_TIP,  HandLM.INDEX_PIP),
            extended(HandLM.MIDDLE_TIP, HandLM.MIDDLE_PIP),
            extended(HandLM.RING_TIP,   HandLM.RING_PIP),
            extended(HandLM.PINKY_TIP,  HandLM.PINKY_PIP),
        ]


    def _palm_normal(self, lm: np.ndarray) -> np.ndarray:
        v1 = lm[HandLM.INDEX_MCP, :3] - lm[HandLM.WRIST, :3]
        v2 = lm[HandLM.PINKY_MCP, :3] - lm[HandLM.WRIST, :3]

        n = np.cross(v1, v2)
        mag = np.linalg.norm(n)

        if mag > 1e-6:
            return (n / mag).astype(np.float32)

        return np.zeros(3, dtype=np.float32)

    def _palm_facing(self, n: np.ndarray) -> str:
        x, y, z = n
        abs_vals = [abs(x), abs(y), abs(z)]
        i = int(np.argmax(abs_vals))

        return [
            "right" if x > 0 else "left",
            "down"  if y > 0 else "up",
            "back"  if z > 0 else "front",
        ][i]


    def _classify_shape(
        self,
        f: list[bool],
        lm: np.ndarray,
    ) -> Optional[str]:

        #  1. PINCH FIRST (fix critique)
        pinch_dist = np.linalg.norm(
            lm[HandLM.THUMB_TIP, :2] - lm[HandLM.INDEX_TIP, :2]
        )
        if (pinch_dist < self.PINCH_THRESHOLD
                and not f[2] and not f[3] and not f[4]):
            return "pinch"

        #  2. Cas simples rapides
        if not any(f):
            return "fist"

        if all(f):
            return "open"

        #  3. Lookup table optimisé
        key = tuple(f)

        shapes = {
            (False, True,  True,  False, False): "peace",
            (False, True,  False, False, False): "point",
            (True,  True,  False, False, False): "gun",
            (False, True,  False, False, True):  "horns",
            (True,  False, False, False, True):  "call",
            (True,  True,  True,  True,  False): "four_fingers",
            (False, True,  True,  True,  True):  "four_fingers",
        }

        return shapes.get(key, None)


    def _compute_velocity(
        self,
        lm_p: np.ndarray,
        prev_hand: Optional[HandData],
        dt: float,
    ) -> tuple[np.ndarray, float]:

        if prev_hand is None or dt <= 1e-6:
            return (
                np.zeros(2, dtype=np.float32),
                0.0,
            )

        delta = (
            lm_p[HandLM.WRIST].astype(np.float64)
            - prev_hand.landmarks_px[HandLM.WRIST].astype(np.float64)
        )

        velocity = (delta / dt).astype(np.float32)
        speed = float(np.linalg.norm(velocity))

        return velocity, speed