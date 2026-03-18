import os
import cv2
import mediapipe as mp
import numpy as np
import time
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

os.environ.setdefault("GLOG_minloglevel",     "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult

from config import AppConfig
from perception.frame_data import FrameData, HandData, PoseData, HandLM, PoseLM


HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
)

MODELS_DIR      = Path("models")
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
POSE_MODEL_PATH = MODELS_DIR / "pose_landmarker_full.task"

HAND_CONNECTIONS: list[tuple[int, int]] = [
    # Pouce
    (HandLM.WRIST,      HandLM.THUMB_CMC),
    (HandLM.THUMB_CMC,  HandLM.THUMB_MCP),
    (HandLM.THUMB_MCP,  HandLM.THUMB_IP),
    (HandLM.THUMB_IP,   HandLM.THUMB_TIP),
    # Index
    (HandLM.WRIST,      HandLM.INDEX_MCP),
    (HandLM.INDEX_MCP,  HandLM.INDEX_PIP),
    (HandLM.INDEX_PIP,  HandLM.INDEX_DIP),
    (HandLM.INDEX_DIP,  HandLM.INDEX_TIP),
    # Majeur
    (HandLM.WRIST,      HandLM.MIDDLE_MCP),
    (HandLM.MIDDLE_MCP, HandLM.MIDDLE_PIP),
    (HandLM.MIDDLE_PIP, HandLM.MIDDLE_DIP),
    (HandLM.MIDDLE_DIP, HandLM.MIDDLE_TIP),
    # Annulaire
    (HandLM.WRIST,      HandLM.RING_MCP),
    (HandLM.RING_MCP,   HandLM.RING_PIP),
    (HandLM.RING_PIP,   HandLM.RING_DIP),
    (HandLM.RING_DIP,   HandLM.RING_TIP),
    # Auriculaire
    (HandLM.WRIST,      HandLM.PINKY_MCP),
    (HandLM.PINKY_MCP,  HandLM.PINKY_PIP),
    (HandLM.PINKY_PIP,  HandLM.PINKY_DIP),
    (HandLM.PINKY_DIP,  HandLM.PINKY_TIP),
    # Transversales paume
    (HandLM.INDEX_MCP,  HandLM.MIDDLE_MCP),
    (HandLM.MIDDLE_MCP, HandLM.RING_MCP),
    (HandLM.RING_MCP,   HandLM.PINKY_MCP),
]

POSE_CONNECTIONS: list[tuple[int, int]] = [
    # Torse
    (PoseLM.LEFT_SHOULDER,  PoseLM.RIGHT_SHOULDER),
    (PoseLM.LEFT_SHOULDER,  PoseLM.LEFT_HIP),
    (PoseLM.RIGHT_SHOULDER, PoseLM.RIGHT_HIP),
    (PoseLM.LEFT_HIP,       PoseLM.RIGHT_HIP),
    # Bras gauche
    (PoseLM.LEFT_SHOULDER,  PoseLM.LEFT_ELBOW),
    (PoseLM.LEFT_ELBOW,     PoseLM.LEFT_WRIST),
    # Bras droit
    (PoseLM.RIGHT_SHOULDER, PoseLM.RIGHT_ELBOW),
    (PoseLM.RIGHT_ELBOW,    PoseLM.RIGHT_WRIST),
    # Jambes
    (PoseLM.LEFT_HIP,       PoseLM.LEFT_KNEE),
    (PoseLM.RIGHT_HIP,      PoseLM.RIGHT_KNEE),
    (PoseLM.LEFT_KNEE,      26),
    (PoseLM.RIGHT_KNEE,     27),
    # Tête
    (PoseLM.NOSE,           PoseLM.LEFT_SHOULDER),
    (PoseLM.NOSE,           PoseLM.RIGHT_SHOULDER),
]


class _Smoother:

    def __init__(self, alpha: float = 0.5, max_age: float = 0.12):
    
        self._alpha   = alpha
        self._max_age = max_age
        self._cache: dict[str, tuple[np.ndarray, float]] = {}

    def smooth(
        self,
        key:       str,
        landmarks: Optional[np.ndarray],
        timestamp: float,
    ) -> Optional[np.ndarray]:
       
        if landmarks is not None:
            if key in self._cache:
                prev_lm, _ = self._cache[key]
                landmarks = (
                    self._alpha * landmarks
                    + (1.0 - self._alpha) * prev_lm
                )
            self._cache[key] = (landmarks.copy(), timestamp)
            return landmarks

        if key in self._cache:
            prev_lm, prev_ts = self._cache[key]
            if timestamp - prev_ts < self._max_age:
                return prev_lm

        self._cache.pop(key, None)
        return None

    def reset(self, key: Optional[str] = None) -> None:
        if key is None:
            self._cache.clear()
        else:
            self._cache.pop(key, None)


def _ensure_model(url: str, dest: Path) -> None:
    if dest.exists():
        return
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[MediaPipe] Téléchargement : {dest.name} ...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"[MediaPipe] Modèle prêt   : {dest}")
    except Exception as e:
        raise RuntimeError(
            f"Impossible de télécharger {dest.name}.\n"
            f"URL    : {url}\n"
            f"Erreur : {e}\n"
            f"Télécharge manuellement et place le fichier dans models/."
        ) from e


class MediaPipeLayer:

    _C_RIGHT_LINE  = (0,   180,  90)  
    _C_RIGHT_POINT = (0,   220, 120)   
    _C_LEFT_LINE   = (200, 120,   0)   
    _C_LEFT_POINT  = (255, 160,   0)
    _C_POSE_LINE   = (80,  200, 120)   
    _C_POSE_POINT  = (80,  200, 120)
    _C_WHITE       = (255, 255, 255)   

    def __init__(self, config: AppConfig):
        self._cfg         = config
        self._frame_index = 0
        self._pose_enabled = False 
        self._ts_ref = int(time.perf_counter() * 1000)

        _ensure_model(HAND_MODEL_URL, HAND_MODEL_PATH)
        _ensure_model(POSE_MODEL_URL, POSE_MODEL_PATH)

        hand_opts = HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(
                model_asset_path=str(HAND_MODEL_PATH)
            ),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=config.mediapipe.min_detection_confidence,
            min_hand_presence_confidence=config.mediapipe.min_tracking_confidence,
            min_tracking_confidence=config.mediapipe.min_tracking_confidence,
        )
        self._hand_landmarker = HandLandmarker.create_from_options(hand_opts)

        pose_opts = PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(
                model_asset_path=str(POSE_MODEL_PATH)
            ),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=config.mediapipe.min_detection_confidence,
            min_pose_presence_confidence=config.mediapipe.min_tracking_confidence,
            min_tracking_confidence=config.mediapipe.min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self._pose_landmarker = PoseLandmarker.create_from_options(pose_opts)

        self._smoother = _Smoother(
            alpha=0.5,     
            max_age=0.12,  
        )

        print(
            f"[MediaPipeLayer] Prêt — MediaPipe {mp.__version__}  "
            f"| détection={config.mediapipe.min_detection_confidence}  "
            f"tracking={config.mediapipe.min_tracking_confidence}"
        )


    def process(self, frame_bgr: np.ndarray) -> FrameData:
        
        h, w = frame_bgr.shape[:2]

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._frame_index += 1

        ts_ms = (
            int(time.perf_counter() * 1000)
            - self._ts_ref
            + self._frame_index
        )

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        now = time.perf_counter()

        hand_result: HandLandmarkerResult = \
            self._hand_landmarker.detect_for_video(mp_image, ts_ms)
        if self._pose_enabled:
            pose_result: PoseLandmarkerResult = \
                self._pose_landmarker.detect_for_video(mp_image, ts_ms)
            pose = self._parse_pose(pose_result, w, h, now)
        else:
            pose = None

        left_hand, right_hand = self._parse_hands(hand_result, w, h, now)

        return FrameData(
            timestamp=now,
            frame_index=self._frame_index,
            frame_rgb=frame_rgb,
            frame_h=h,
            frame_w=w,
            left_hand=left_hand,
            right_hand=right_hand,
            pose=pose,
        )

    def process_with_draw(
        self,
        frame_bgr: np.ndarray,
    ) -> Tuple[FrameData, np.ndarray]:
       
        frame_data = self.process(frame_bgr)
        annotated  = self.draw_landmarks(frame_bgr, frame_data)
        return frame_data, annotated

    def draw_landmarks(
        self,
        frame_bgr:  np.ndarray,
        frame_data: FrameData,
    ) -> np.ndarray:
       
        out = frame_bgr.copy()

        if frame_data.left_hand is not None:
            self._draw_hand(
                out,
                frame_data.left_hand.landmarks_px,
                self._C_LEFT_LINE,
                self._C_LEFT_POINT,
            )

        if frame_data.right_hand is not None:
            self._draw_hand(
                out,
                frame_data.right_hand.landmarks_px,
                self._C_RIGHT_LINE,
                self._C_RIGHT_POINT,
            )

        if frame_data.pose is not None:
            self._draw_pose(out, frame_data.pose)

        return out

    def set_smoothing(self, alpha: float, max_age: float) -> None:
        
        self._smoother._alpha   = max(0.0, min(1.0, alpha))
        self._smoother._max_age = max(0.0, max_age)
        self._smoother.reset()

    def enable_pose(self) -> None:
        self._pose_enabled = True
        self._smoother.reset("pose")
        print("[MediaPipePlayer] Pose activée")
    
    def disable_pose(self) -> None:
        self._pose_enabled = False
        self._smoother.reset("pose")
        print("[MediaPipeLayer] Pose désactivée")
    
    def toggle_pose(self) -> bool:
        if self._pose_enabled:
            self.disable_pose()
        else:
            self.enable_pose()
        return self._pose_enabled
    
    @property
    def pose_enabled(self) -> bool:
        return self._pose_enabled

    def release(self) -> None:
        self._hand_landmarker.close()
        self._pose_landmarker.close()
        print("[MediaPipeLayer] Ressources libérées.")


    def _parse_hands(
        self,
        result:    HandLandmarkerResult,
        w:         int,
        h:         int,
        timestamp: float,
    ) -> Tuple[Optional[HandData], Optional[HandData]]:
        
        raw: dict[str, tuple[np.ndarray, float]] = {}

        if result.hand_landmarks:
            for lm_list, handedness_list in zip(
                result.hand_landmarks, result.handedness
            ):
                side = handedness_list[0].category_name   
                if self._cfg.camera.flip_horizontal:
                    side = "Right" if side == "Left" else "Left"

                lm_norm = np.array(
                    [[lm.x, lm.y, lm.z] for lm in lm_list],
                    dtype=np.float32,
                )   # (21, 3)
                raw[side] = (lm_norm, float(handedness_list[0].score))

        def _build(side: str) -> Optional[HandData]:
            lm_raw, score = raw.get(side, (None, 0.0))
            smoothed = self._smoother.smooth(side.lower(), lm_raw, timestamp)
            if smoothed is None:
                return None
            return HandData(
                landmarks=smoothed,
                landmarks_px=self._norm_to_px(smoothed, w, h),
                handedness=side,
                score=score if lm_raw is not None else 0.0,
            )

        return _build("Left"), _build("Right")

    def _parse_pose(
        self,
        result:    PoseLandmarkerResult,
        w:         int,
        h:         int,
        timestamp: float,
    ) -> Optional[PoseData]:
        lm_raw: Optional[np.ndarray] = None

        if result.pose_landmarks:
            lm_list = result.pose_landmarks[0]   
            lm_raw  = np.array(
                [[lm.x, lm.y, lm.z, lm.visibility] for lm in lm_list],
                dtype=np.float32,
            )   # (33, 4)

        xyz_raw      = lm_raw[:, :3] if lm_raw is not None else None
        xyz_smoothed = self._smoother.smooth("pose", xyz_raw, timestamp)

        if xyz_smoothed is None:
            return None

        visibility = (
            lm_raw[:, 3:4]
            if lm_raw is not None
            else np.full((33, 1), 0.5, dtype=np.float32)
        )
        pts   = np.concatenate([xyz_smoothed, visibility], axis=1)  
        lm_px = self._norm_to_px(xyz_smoothed, w, h)                 

        key_indices = [
            PoseLM.LEFT_SHOULDER, PoseLM.RIGHT_SHOULDER,
            PoseLM.LEFT_ELBOW,    PoseLM.RIGHT_ELBOW,
        ]
        upper_visible = all(pts[i, 3] > 0.4 for i in key_indices)

        return PoseData(
            landmarks=pts,
            landmarks_px=lm_px,
            upper_body_visible=upper_visible,
        )


    def _draw_hand(
        self,
        frame:       np.ndarray,
        pts:         np.ndarray,   
        line_color:  tuple,
        point_color: tuple,
    ) -> None:
        
        for (a, b) in HAND_CONNECTIONS:
            cv2.line(
                frame,
                (int(pts[a, 0]), int(pts[a, 1])),
                (int(pts[b, 0]), int(pts[b, 1])),
                line_color, 1, cv2.LINE_AA,
            )
        for pt in pts:
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(frame, (x, y), 5, point_color,   -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 5, self._C_WHITE,  1, cv2.LINE_AA)

    def _draw_pose(self, frame: np.ndarray, pose: PoseData) -> None:
        pts = pose.landmarks_px   
        lm  = pose.landmarks      

        for (a, b) in POSE_CONNECTIONS:
            if a >= len(pts) or b >= len(pts):
                continue
            if lm[a, 3] < 0.4 or lm[b, 3] < 0.4:
                continue
            cv2.line(
                frame,
                (int(pts[a, 0]), int(pts[a, 1])),
                (int(pts[b, 0]), int(pts[b, 1])),
                self._C_POSE_LINE, 2, cv2.LINE_AA,
            )

        for i, pt in enumerate(pts):
            if lm[i, 3] < 0.4:
                continue
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(frame, (x, y), 5, self._C_POSE_POINT, -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 5, self._C_WHITE,        1, cv2.LINE_AA)


    @staticmethod
    def _norm_to_px(lm: np.ndarray, w: int, h: int) -> np.ndarray:
       
        return np.stack([
            np.clip(lm[:, 0] * w, 0, w - 1).astype(np.int32),
            np.clip(lm[:, 1] * h, 0, h - 1).astype(np.int32),
        ], axis=1)