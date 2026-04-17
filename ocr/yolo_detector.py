from __future__ import annotations

import os
import numpy as np
from typing import List, Tuple, Optional

BBox = Tuple[int, int, int, int]

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "bubble_detector.pt")
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45


class YOLOBubbleDetector:
    """
    Детекция баблов манги через YOLOv8.
    Если detector_backend=yolo и модель найдена — YOLO.
    Если detector_backend=opencv или модель не найдена — OpenCV BubbleDetector.
    """

    def __init__(self, model_path: Optional[str] = None, debug_path: Optional[str] = None):
        self._model_path = model_path or MODEL_PATH
        self._debug_path = debug_path
        self._model = None
        self._loaded = False
        self._fallback = None

        from config.settings import settings
        self._force_fallback = settings.detector_backend.value == "opencv"
        if settings.yolo_model_path and not model_path:
            self._model_path = settings.yolo_model_path

    def _try_load(self) -> bool:
        if self._loaded:
            return self._model is not None

        self._loaded = True

        if self._force_fallback:
            print("[YOLO] Настройка: detector_backend=opencv → OpenCV BubbleDetector")
            from ocr.bubble_detector import BubbleDetector

            self._fallback = BubbleDetector(debug_path=self._debug_path)
            return False

        if not os.path.exists(self._model_path):
            print(f"[YOLO] Модель не найдена: {self._model_path}")
            print("[YOLO] Fallback → OpenCV BubbleDetector")
            from ocr.bubble_detector import BubbleDetector

            self._fallback = BubbleDetector(debug_path=self._debug_path)
            return False

        try:
            from ultralytics import YOLO

            print(f"[YOLO] Загрузка модели: {self._model_path}")
            self._model = YOLO(self._model_path)
            print("[YOLO] Модель загружена")
            return True
        except Exception as e:
            print(f"[YOLO] Ошибка загрузки: {e}")
            print("[YOLO] Fallback → OpenCV BubbleDetector")
            from ocr.bubble_detector import BubbleDetector

            self._fallback = BubbleDetector(debug_path=self._debug_path)
            return False

    def detect(self, image: np.ndarray) -> List[BBox]:
        """
        Детекция баблов. Возвращает список (x, y, w, h).
        YOLO → если нет модели → OpenCV fallback.
        """
        if not self._loaded:
            self._try_load()

        if self._model is not None:
            return self._detect_yolo(image)

        if self._fallback is not None:
            return self._fallback.detect(image)

        return []

    def detect_from_text(
        self,
        image: np.ndarray,
        text_regions: List[Tuple[BBox, str]],
    ) -> List[BBox]:
        """
        Текстовая стратегия (fallback к OpenCV).
        YOLO не использует текстовые регионы — он визуальный.
        """
        if not self._loaded:
            self._try_load()

        if self._fallback is not None:
            return self._fallback.detect_from_text(image, text_regions)

        if self._model is not None:
            return self._detect_yolo(image)

        return []

    def _detect_yolo(self, image: np.ndarray) -> List[BBox]:
        try:
            results = self._model(
                image,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                verbose=False,
            )

            bubbles = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                    if w < 8 or h < 6:
                        continue
                    bubbles.append((x, y, w, h))

            print(f"[YOLO] Найдено {len(bubbles)} баблов")
            return bubbles

        except Exception as e:
            print(f"[YOLO] Ошибка детекции: {e}")
            return []
