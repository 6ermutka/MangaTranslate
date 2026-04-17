from __future__ import annotations

import textwrap
import threading
import time
from typing import Callable, Optional, Tuple, List
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from ocr.ocr_engine import OCREngine
from translation.translator import Translator
from ocr.yolo_detector import YOLOBubbleDetector
from config.settings import settings

FONT_PATH = "/Users/stepanivanov/Documents/MangaTranslate/fonts/animeacev05.ttf"
FALLBACK_FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
FALLBACK_FONT_PATH_2 = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

CHAR_FIXES = {
    '♡': '♥',
    '☆': '★',
    '♪': '♪',
}


def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    for p in [path, FALLBACK_FONT_PATH, FALLBACK_FONT_PATH_2]:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def process_contour(crop: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mean_val = int(gray.mean())

    if mean_val < 128:
        white_val = 0
        fg_val = 255
        fill_color = (0, 0, 0)
        text_color_fill = (0, 0, 0)
    else:
        white_val = 255
        fg_val = 0
        fill_color = (255, 255, 255)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if mean_val < 128:
        thresh = cv2.bitwise_not(thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        result = crop.copy()
        h, w = result.shape[:2]
        result[:] = fill_color
        dummy = np.array([[0, 0], [w, 0], [w, h], [0, h]])
        return result, dummy.reshape(-1, 1, 2)

    largest = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest], -1, 255, cv2.FILLED)

    result = crop.copy()
    result[mask == 255] = fill_color

    return result, largest


def _wrap_with_hyphen(text: str, font: ImageFont.FreeTypeFont, max_w: int) -> List[str]:
    words = text.split()
    if not words:
        return [text]
    lines = []
    cur = ""
    for word in words:
        test = f"{cur} {word}".strip()
        bw = font.getbbox(test)[2] - font.getbbox(test)[0]
        if bw <= max_w:
            cur = test
        else:
            if cur:
                lines.append(cur)
            word_w = font.getbbox(word)[2] - font.getbbox(word)[0]
            if word_w > max_w:
                chunk = ""
                for ch in word:
                    test_ch = chunk + ch
                    cw = font.getbbox(test_ch)[2] - font.getbbox(test_ch)[0]
                    hyph_w = font.getbbox(chunk + "-")[2] - font.getbbox(chunk + "-")[0]
                    if hyph_w > max_w:
                        if chunk:
                            lines.append(chunk + "-")
                        chunk = ch
                    elif cw > max_w:
                        if len(chunk) > 1:
                            lines.append(chunk + "-")
                            chunk = ch
                        else:
                            chunk = ch
                    else:
                        chunk = test_ch
                cur = chunk
            else:
                cur = word
    if cur:
        lines.append(cur)
    if not lines:
        lines = [text]
    return lines


def render_text(
    canvas_pil: Image.Image,
    x1: int, y1: int, x2: int, y2: int,
    text: str,
    contour: Optional[np.ndarray] = None,
    is_dark: bool = False,
):
    pad = 6
    if contour is not None:
        cx, cy, cw, ch = cv2.boundingRect(contour)
        rx1 = x1 + cx + pad
        ry1 = y1 + cy + pad
        rx2 = x1 + cx + cw - pad
        ry2 = y1 + cy + ch - pad
    else:
        rx1 = x1 + pad
        ry1 = y1 + pad
        rx2 = x2 - pad
        ry2 = y2 - pad

    inner_w = rx2 - rx1
    inner_h = ry2 - ry1
    if inner_w <= 0 or inner_h <= 0 or not text.strip():
        return

    text = text.upper()

    line_height = 18
    font_size = 19
    wrapping_ratio = 0.075
    min_font_size = 10
    max_iterations = 30

    draw = ImageDraw.Draw(canvas_pil)

    for iteration in range(max_iterations):
        font = _load_font(FONT_PATH, font_size)
        line_height = max(font_size + 4, 12)

        lines = _wrap_with_hyphen(text, font, inner_w)

        total_h = len(lines) * line_height

        if total_h <= inner_h:
            break

        font_size = max(font_size - 1, min_font_size)
    else:
        max_lines = max(1, inner_h // line_height)
        lines = lines[:max_lines]
        if lines:
            lines[-1] = lines[-1][:max(0, len(lines[-1]) - 3)] + "..."

    actual_h = len(lines) * line_height
    text_y = ry1 + max(0, (inner_h - actual_h) // 2)

    for line in lines:
        try:
            text_length = draw.textlength(line, font=font)
        except Exception:
            text_length = len(line) * font_size * 0.6

        text_x = rx1 + max(0, (inner_w - text_length) // 2)

        if is_dark:
            outline_fill = (0, 0, 0, 200)
            text_fill = (255, 255, 255, 255)
        else:
            outline_fill = (255, 255, 255, 200)
            text_fill = (0, 0, 0, 255)

        for ox, oy in [(-1,-1),(-1,1),(1,-1),(1,1),(0,-1),(0,1),(-1,0),(1,0)]:
            draw.text((text_x+ox, text_y+oy), line, font=font, fill=outline_fill)
        draw.text((text_x, text_y), line, font=font, fill=text_fill)

        text_y += line_height


class TranslationPipeline:
    """
    Оффлайн пайплайн: захват → YOLO → OCR → заливка белым → перевод → рендер.
    """

    def __init__(self, on_result: Callable[[np.ndarray], None], scale_factor: float = 1.0):
        self._on_result = on_result
        self._scale_factor = scale_factor

        from capture.screen_capture import ScreenCapture
        self._capture = ScreenCapture(scale_factor=scale_factor)
        self._ocr = OCREngine()
        self._translator = Translator()
        self._ready = False

    def warmup(self):
        print("[Pipeline] Прогрев моделей...")
        self._ocr.warmup(settings.source_lang)
        self._ready = True
        print("[Pipeline] Готов")

    def models_ready(self) -> bool:
        from translation.model_manager import all_required_downloaded
        return all_required_downloaded(settings.source_lang, settings.target_lang)

    def snap(self, zone: Tuple[int, int, int, int]):
        x, y, w, h = zone
        t0 = time.monotonic()

        frame = self._capture.capture(x, y, w, h)
        img_h, img_w = frame.shape[:2]

        t1 = time.monotonic()
        src = settings.source_lang or "auto"
        ocr_results = self._ocr.detect(frame, src)
        n = len(ocr_results)
        print(f"[Pipeline] OCR: {n} баблов за {(time.monotonic()-t1)*1000:.0f}мс")

        if n == 0:
            print("[Pipeline] Баблы не найдены")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._on_result(rgb)
            return

        canvas_bgr = frame.copy()
        regions_to_translate = []
        box_data = []

        for i, (bbox, orig) in enumerate(ocr_results):
            bx, by, bw, bh = bbox
            x1 = max(0, bx)
            y1 = max(0, by)
            x2 = min(img_w, bx + bw)
            y2 = min(img_h, by + bh)
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            cleaned, contour = process_contour(crop)
            canvas_bgr[y1:y2, x1:x2] = cleaned
            is_dark = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean() < 128

            if orig:
                regions_to_translate.append((bbox, orig))
                box_data.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'orig': orig, 'contour': contour, 'is_dark': is_dark,
                })

        t3 = time.monotonic()
        translated_regions = []
        if regions_to_translate:
            translated_regions = self._translator.translate_regions(
                regions_to_translate,
                source_lang=settings.source_lang,
                target_lang=settings.target_lang,
            )
        print(f"[Pipeline] Перевод: {(time.monotonic()-t3)*1000:.0f}мс")

        translated_map = {}
        for bbox, orig, translated in translated_regions:
            translated_map[bbox] = translated

        canvas_pil = Image.fromarray(cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB))
        for data in box_data:
            x1, y1, x2, y2 = data['x1'], data['y1'], data['x2'], data['y2']
            bbox = (x1, y1, x2 - x1, y2 - y1)
            translated = translated_map.get(bbox, "")
            if translated:
                for old, new in CHAR_FIXES.items():
                    translated = translated.replace(old, new)
                render_text(canvas_pil, x1, y1, x2, y2, translated,
                            contour=data['contour'], is_dark=data['is_dark'])

        result_rgb = np.array(canvas_pil)
        total = (time.monotonic() - t0) * 1000
        print(f"[Pipeline] Готово за {total:.0f}мс")
        self._on_result(result_rgb)
