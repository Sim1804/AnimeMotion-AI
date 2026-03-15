from dataclasses import dataclass, field
from typing import Literal, Optional

@dataclass
class CameraConfig:
    device_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    flip_horizontal: bool = True
    autofocus: bool = False
    auto_exposure: bool = True

@dataclass
class MediaPipeConfig:
    min_detection_confidence: float = 0.3
    min_tracking_confidence: float = 0.3
    model_complexity: int = 1
    smooth_landmarks: bool = True

@dataclass
class UniverseConfig:
    universes_dir: str = "techniques/universes"
    hot_reload: bool = True
    active_universes: Optional[list] = None
    player_level: int = 1
    energy_max: float = 100.0

@dataclass
class AppConfig:
    mode: Literal["desktop", "combat"] = "desktop"
    camera: CameraConfig = field(default_factory=CameraConfig)
    mediapipe: MediaPipeConfig = field(default_factory=MediaPipeConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    debug_overlay: bool = True
    data_collection_mode: bool = False

CONFIG = AppConfig()