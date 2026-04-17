from __future__ import annotations

import cv2
import numpy as np
from typing import List, Tuple, Optional

BBox = Tuple[int, int, int, int]


class BubbleDetector:
    """
    Две стратегии поиска баблов в манге:
    1. Визуальная: Otsu → dilate тёмных областей → flood fill фона → контуры светлых остатков
    2. Текстовая: кластеризация OCR-фрагментов → консервативное расширение до тёмной обводки
    """

    def __init__(self, debug_path: Optional[str] = None):
        self._debug_path = debug_path

    # ── Стратегия 1: визуальная ────────────────────────────────────

    def detect(self, image: np.ndarray) -> List[BBox]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_h, img_w = gray.shape
        img_area = img_h * img_w

        _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        self._dbg("1_otsu_dark.png", dark)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dark_dilated = cv2.dilate(dark, kernel, iterations=3)

        self._dbg("2_dilated.png", dark_dilated)

        light = cv2.bitwise_not(dark_dilated)

        light_no_bg = self._remove_background(light, img_h, img_w)

        self._dbg("3_no_bg.png", light_no_bg)

        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(light_no_bg, cv2.MORPH_OPEN, kernel_open, iterations=1)

        kernel_close2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close2, iterations=1)

        self._dbg("4_cleaned.png", cleaned)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bubbles = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            rect_area = w * h
            ratio = rect_area / img_area
            if ratio < 0.003 or ratio > 0.35:
                continue
            aspect = w / h if h > 0 else 0
            if aspect < 0.25 or aspect > 4.0:
                continue

            interior = gray[y:y + h, x:x + w]
            if interior.size == 0 or interior.mean() < 120:
                continue

            dark_ratio = (interior < 80).sum() / interior.size
            if dark_ratio < 0.002:
                continue

            bubbles.append((x, y, w, h))

        return self._nms(bubbles)

    # ── Стратегия 2: текстовая ─────────────────────────────────────

    def detect_from_text(
        self,
        image: np.ndarray,
        text_regions: List[Tuple[BBox, str]],
    ) -> List[BBox]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_h, img_w = gray.shape

        clusters = self._cluster_text(text_regions)

        bubbles = []
        for cluster in clusters:
            xs = [r[0][0] for r in cluster]
            ys = [r[0][1] for r in cluster]
            xs2 = [r[0][0] + r[0][2] for r in cluster]
            ys2 = [r[0][1] + r[0][3] for r in cluster]

            tx1, ty1 = min(xs), min(ys)
            tx2, ty2 = max(xs2), max(ys2)
            tw, th = tx2 - tx1, ty2 - ty1

            bx1, by1, bx2, by2 = self._expand_to_border(
                gray, tx1, ty1, tx2, ty2, img_h, img_w,
            )
            bw, bh = bx2 - bx1, by2 - by1

            if bw > tw * 3.0 or bh > th * 3.0:
                print(f"    [BD] Расширение слишком большое ({bw}x{bh} vs текст {tw}x{th}), обрезаю")
                bx1 = tx1 - int(tw * 0.4)
                by1 = ty1 - int(th * 0.4)
                bx2 = tx2 + int(tw * 0.4)
                by2 = ty2 + int(th * 0.4)
                bx1 = max(0, bx1)
                by1 = max(0, by1)
                bx2 = min(img_w, bx2)
                by2 = min(img_h, by2)

            bubbles.append((bx1, by1, bx2 - bx1, by2 - by1))

        return self._nms(bubbles)

    # ── Внутренние методы ──────────────────────────────────────────

    def _remove_background(self, light, img_h, img_w):
        flood = light.copy()
        mask = np.zeros((img_h + 2, img_w + 2), np.uint8)

        step = max(1, min(img_h, img_w) // 150)
        for i in range(0, img_h, step):
            if flood[i, 0] == 255:
                cv2.floodFill(flood, mask, (0, i), 0)
            if flood[i, img_w - 1] == 255:
                cv2.floodFill(flood, mask, (img_w - 1, i), 0)
        for i in range(0, img_w, step):
            if flood[0, i] == 255:
                cv2.floodFill(flood, mask, (i, 0), 0)
            if flood[img_h - 1, i] == 255:
                cv2.floodFill(flood, mask, (i, img_h - 1), 0)

        return flood

    def _expand_to_border(self, gray, x1, y1, x2, y2, img_h, img_w):
        interior = gray[max(0, y1):min(img_h, y2),
                        max(0, x1):min(img_w, x2)]
        if interior.size == 0:
            return x1, y1, x2, y2

        ref_mean = interior.mean()
        border_thresh = max(60, ref_mean * 0.55)

        tw, th = x2 - x1, y2 - y1
        max_x_expand = int(tw * 0.5)
        max_y_expand = int(th * 0.5)

        pad = 10
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(img_w, x2 + pad)
        y2 = min(img_h, y2 + pad)

        check_w = 6

        for _ in range(max_x_expand):
            if x1 <= 0:
                break
            nx = max(0, x1 - check_w)
            col = gray[y1:y2, nx:x1].flatten()
            if col.size == 0:
                break
            if col.mean() < border_thresh:
                break
            x1 = nx

        for _ in range(max_x_expand):
            if x2 >= img_w:
                break
            nx2 = min(img_w, x2 + check_w)
            col = gray[y1:y2, x2:nx2].flatten()
            if col.size == 0:
                break
            if col.mean() < border_thresh:
                break
            x2 = nx2

        for _ in range(max_y_expand):
            if y1 <= 0:
                break
            ny = max(0, y1 - check_w)
            row = gray[ny:y1, x1:x2].flatten()
            if row.size == 0:
                break
            if row.mean() < border_thresh:
                break
            y1 = ny

        for _ in range(max_y_expand):
            if y2 >= img_h:
                break
            ny2 = min(img_h, y2 + check_w)
            row = gray[y2:ny2, x1:x2].flatten()
            if row.size == 0:
                break
            if row.mean() < border_thresh:
                break
            y2 = ny2

        margin = 5
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(img_w, x2 + margin)
        y2 = min(img_h, y2 + margin)

        return x1, y1, x2, y2

    def _cluster_text(self, text_regions, gap_y=18, gap_x=45):
        if not text_regions:
            return []
        sorted_r = sorted(text_regions, key=lambda r: (r[0][1], r[0][0]))
        clusters: List[List] = []
        current = [sorted_r[0]]

        for reg in sorted_r[1:]:
            fx, fy, fw, fh = reg[0]
            near = False
            for (gx, gy, gw, gh), _ in current:
                dy = abs(fy - (gy + gh))
                x_near = fx < gx + gw + gap_x and fx + fw > gx - gap_x
                if dy < gap_y and x_near:
                    near = True
                    break
            if near:
                current.append(reg)
            else:
                clusters.append(current)
                current = [reg]
        clusters.append(current)
        return clusters

    def _dbg(self, name, img):
        if self._debug_path:
            cv2.imwrite(f"{self._debug_path}_{name}", img)

    def _nms(self, bubbles: List[BBox], thresh: float = 0.5) -> List[BBox]:
        if not bubbles:
            return []
        sorted_b = sorted(bubbles, key=lambda b: b[2] * b[3], reverse=True)
        kept = []
        for b in sorted_b:
            if not any(self._overlap_ratio(b, k) > thresh for k in kept):
                kept.append(b)
        return kept

    @staticmethod
    def _overlap_ratio(a: BBox, b: BBox) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix = max(ax, bx);  iy = max(ay, by)
        ix2 = min(ax + aw, bx + bw);  iy2 = min(ay + ah, by + bh)
        if ix2 <= ix or iy2 <= iy:
            return 0.0
        inter = (ix2 - ix) * (iy2 - iy)
        smaller = min(aw * ah, bw * bh)
        return inter / smaller if smaller > 0 else 0.0
