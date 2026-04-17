from dataclasses import dataclass, field
from enum import Enum


class DetectorBackend(str, Enum):
    YOLO   = "yolo"
    OPENCV = "opencv"


@dataclass
class AppSettings:
    source_lang: str = "auto"
    target_lang: str = "ru"

    detector_backend: DetectorBackend = DetectorBackend.YOLO
    yolo_model_path: str = ""

    overlay_x: int = 100
    overlay_y: int = 100
    overlay_width: int = 800
    overlay_height: int = 600

    font_size_max: int = 22
    font_size_min: int = 8
    text_padding: int = 6

    cache_max_size: int = 200

    user_dict: dict = field(default_factory=lambda: {
        "-tan": "-тян",
        "-kun": "-кун",
        "-chan": "-тян",
        "-san": "-сан",
        "-sama": "-сама",
        "-senpai": "-сэмпай",
        "-sensei": "-сэнсэй",
    })


settings = AppSettings()
