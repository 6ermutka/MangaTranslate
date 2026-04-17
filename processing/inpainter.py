import cv2
import numpy as np
from typing import List, Tuple


BBox = Tuple[int, int, int, int]  # x, y, w, h


class Inpainter:
    """
    Закрашивает области с оригинальным текстом.
    MVP: белый прямоугольник с небольшим padding.
    """

    PADDING = 4  # пикселей отступа вокруг bbox

    def erase_regions(self, image: np.ndarray, bboxes: List[BBox]) -> np.ndarray:
        """
        Принимает BGR изображение и список bbox.
        Возвращает изображение с закрашенными регионами.
        """
        result = image.copy()
        for (x, y, w, h) in bboxes:
            x1 = max(0, x - self.PADDING)
            y1 = max(0, y - self.PADDING)
            x2 = min(image.shape[1], x + w + self.PADDING)
            y2 = min(image.shape[0], y + h + self.PADDING)
            cv2.rectangle(result, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1)

        return result

    def erase_with_inpaint(self, image: np.ndarray, bboxes: List[BBox]) -> np.ndarray:
        """
        Улучшенное стирание через OpenCV inpaint (восстанавливает фон).
        Медленнее, но выглядит чище.
        """
        result = image.copy()
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        for (x, y, w, h) in bboxes:
            x1 = max(0, x - self.PADDING)
            y1 = max(0, y - self.PADDING)
            x2 = min(image.shape[1], x + w + self.PADDING)
            y2 = min(image.shape[0], y + h + self.PADDING)
            mask[y1:y2, x1:x2] = 255

        if mask.any():
            result = cv2.inpaint(result, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        return result
