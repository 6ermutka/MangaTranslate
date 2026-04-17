import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple
from config.settings import settings


BBox = Tuple[int, int, int, int]
TranslatedRegion = Tuple[BBox, str, str]  # (bbox, original, translated)

BUBBLE_PADDING = 6   # отступ белого фона вокруг текста
BUBBLE_RADIUS  = 6   # скругление углов пузырька


class TextOverlay:
    """
    Рисует переведённые пузырьки на прозрачном холсте (RGBA).
    Везде где нет текста — полностью прозрачно.
    """

    FONT_PATH          = "/System/Library/Fonts/Supplemental/Arial.ttf"
    FALLBACK_FONT_PATH = "/System/Library/Fonts/Helvetica.ttc"

    def __init__(self):
        self._font_cache = {}

    # ── Главный метод ─────────────────────────────────────────────

    def draw_on_transparent(
        self,
        height: int,
        width: int,
        regions: List[TranslatedRegion],
    ) -> np.ndarray:
        """
        Создаёт прозрачное RGBA изображение размером (height x width).
        Рисует белые пузырьки с переведённым текстом только там где найден текст.
        Возвращает numpy RGBA array для отображения в OverlayWindow.
        """
        # Полностью прозрачный холст
        canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)

        for (bbox, original, translated) in regions:
            x, y, w, h = bbox
            pad = BUBBLE_PADDING

            # Белый фон пузырька со скруглёнными углами
            bx0 = max(0, x - pad)
            by0 = max(0, y - pad)
            bx1 = min(width,  x + w + pad)
            by1 = min(height, y + h + pad)
            self._draw_rounded_rect(draw, bx0, by0, bx1, by1, BUBBLE_RADIUS)

            # Текст перевода внутри пузырька
            inner_w = bx1 - bx0 - pad * 2
            inner_h = by1 - by0 - pad * 2
            self._draw_text_in_box(
                draw, translated,
                bx0 + pad, by0 + pad,
                inner_w, inner_h,
            )

        return np.array(canvas)  # RGBA numpy array

    # ── Рисование ─────────────────────────────────────────────────

    def _draw_rounded_rect(
        self,
        draw: ImageDraw.Draw,
        x0: int, y0: int, x1: int, y1: int,
        radius: int,
        fill=(255, 255, 255, 240),
        outline=(180, 180, 180, 200),
    ):
        draw.rounded_rectangle(
            [x0, y0, x1, y1],
            radius=radius,
            fill=fill,
            outline=outline,
            width=1,
        )

    def _draw_text_in_box(
        self,
        draw: ImageDraw.Draw,
        text: str,
        x: int, y: int,
        max_width: int, max_height: int,
    ):
        if max_width <= 0 or max_height <= 0 or not text.strip():
            return

        for font_size in range(settings.font_size_max, settings.font_size_min - 1, -1):
            font = self._get_font(font_size)
            wrapped = self._wrap_text(text, font, max_width)
            total_height = self._text_block_height(wrapped, font)

            if total_height <= max_height:
                cur_y = y
                for line in wrapped:
                    draw.text((x, cur_y), line, font=font, fill=(10, 10, 10, 255))
                    line_h = font.getbbox(line)[3] - font.getbbox(line)[1]
                    cur_y += line_h + 2
                return

        # Минимальный шрифт — всё равно рисуем
        font = self._get_font(settings.font_size_min)
        wrapped = self._wrap_text(text, font, max_width)
        cur_y = y
        for line in wrapped:
            if cur_y >= y + max_height:
                break
            draw.text((x, cur_y), line, font=font, fill=(10, 10, 10, 255))
            line_h = font.getbbox(line)[3] - font.getbbox(line)[1]
            cur_y += line_h + 2

    # ── Утилиты ───────────────────────────────────────────────────

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        if size not in self._font_cache:
            try:
                self._font_cache[size] = ImageFont.truetype(self.FONT_PATH, size)
            except Exception:
                try:
                    self._font_cache[size] = ImageFont.truetype(self.FALLBACK_FONT_PATH, size)
                except Exception:
                    self._font_cache[size] = ImageFont.load_default()
        return self._font_cache[size]

    def _wrap_text(self, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
        words = text.split()
        lines, current = [], ""
        for word in words:
            test = f"{current} {word}".strip()
            if font.getbbox(test)[2] - font.getbbox(test)[0] <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines or [text]

    def _text_block_height(self, lines: List[str], font: ImageFont.FreeTypeFont) -> int:
        total = 0
        for line in lines:
            b = font.getbbox(line)
            total += (b[3] - b[1]) + 2
        return total
