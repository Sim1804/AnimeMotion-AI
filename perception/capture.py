import cv2
import numpy as np
from config import AppConfig
from typing import Optional

class CameraCapture:

    def __init__(self, config: AppConfig):
        self._cfg = config.camera
        self._cap: cv2.VideoCapture = None
        self._open()

    def _open(self):
        self._cap = cv2.VideoCapture(self._cfg.device_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la caméra idnex={self._cfg.device_index}," "Vérifier les permissions ou essaie un autre index")
        
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.height)
        self._cap.set(cv2.CAP_PROP_FPS, self._cfg.fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self) -> Optional[np.ndarray]:
        ret, frame = self._cap.read()
        if not ret:
            return None
        if self._cfg.flip_horizontal:
            frame = cv2.flip(frame, 1)
        return frame
    
    @property
    def resolution(self) -> tuple[int, int]:
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h
    
    def release(self):
        if self._cap:
            self._cap.release()
