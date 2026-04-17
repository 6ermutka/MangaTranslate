from __future__ import annotations

import base64
import re
from typing import List, Tuple
import numpy as np
import cv2
import requests

from pathlib import Path

from ocr.yolo_detector import YOLOBubbleDetector
from config.settings import settings

TextRegion = Tuple[Tuple[int, int, int, int], str]

BUBBLE_PADDING = 6

FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
VISION_MODEL = "accounts/fireworks/models/qwen3p6-plus"


def _get_api_key() -> str:
    try:
        import json as _j
        cfg = _j.load(open(Path(__file__).parent.parent / "config" / "web_config.json"))
        key = cfg.get("fireworks_api_key", "")
        if key and not key.startswith("YOUR_"):
            return key
    except Exception:
        pass
    import os
    return os.getenv("FIREWORKS_API_KEY", "")

MAX_IMG_SIDE = 1280
JPEG_QUALITY = 78


def _encode_image(image: np.ndarray) -> str:
    h, w = image.shape[:2]
    if max(h, w) > MAX_IMG_SIDE:
        scale = MAX_IMG_SIDE / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return base64.b64encode(buf.tobytes()).decode()


class OCREngine:
    def __init__(self):
        self._detector = YOLOBubbleDetector()

    def warmup(self, source_lang: str = "en"):
        pass

    def detect(self, image: np.ndarray, source_lang: str = "auto") -> List[TextRegion]:
        try:
            bubble_bboxes = self._detector.detect(image)
            print(f"[OCR] YOLO: {len(bubble_bboxes)} bubbles")
            if not bubble_bboxes:
                return []
            return self._ocr_only(image, bubble_bboxes, source_lang)
        except Exception as e:
            import traceback
            print(f"[OCR] Error: {e}")
            traceback.print_exc()
            return []

    def _ocr_only(self, image: np.ndarray, bboxes, source_lang: str = "en") -> List[TextRegion]:
        img_h, img_w = image.shape[:2]
        num = len(bboxes)

        lang_name = {"en": "English", "id": "Indonesian", "ja": "Japanese"}.get(source_lang, "English")

        annotated = image.copy()
        for i, (bx, by, bw, bh) in enumerate(bboxes):
            px = max(0, bx - BUBBLE_PADDING)
            py = max(0, by - BUBBLE_PADDING)
            px2 = min(img_w, bx + bw + BUBBLE_PADDING)
            py2 = min(img_h, by + bh + BUBBLE_PADDING)
            cv2.rectangle(annotated, (px, py), (px2, py2), (0, 255, 0), 2)
            cv2.putText(annotated, str(i + 1), (px + 4, py + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        b64 = _encode_image(annotated)

        prompt = (
            f"Read the {lang_name} text inside each of the {num} numbered green boxes in this manga image.\n"
            f"Also identify who is speaking (based on bubble tail direction, character position/appearance).\n"
            f"Output EXACTLY {num} lines.\n"
            f"Format: NUMBER. [speaker] text\n"
            f"Speaker options: girl, boy, man, woman, child, narrator, unknown\n"
            f"Rules:\n"
            f"- Write ONLY the text as it appears, verbatim. Do NOT translate.\n"
            f"- Identify speaker from visual cues: bubble tail points to character.\n"
            f"- If speaker is unclear, use [unknown].\n"
            f"- Do NOT explain or reason. Start line 1 with '1.'\n"
            f"- If a box is empty or unreadable, write: NUMBER. [unknown] [unreadable]"
        )

        for attempt in range(3):
            if attempt > 0:
                import time as _time
                _time.sleep(3 * attempt)
            try:
                from pathlib import Path
                import json as _json
                cfg_path = Path(__file__).parent.parent / "config" / "web_config.json"
                ocr_system = None
                if cfg_path.exists():
                    with open(cfg_path) as _f:
                        _cfg = _json.load(_f)
                    ocr_system = _cfg.get("ocr_system_prompt")

                if not ocr_system:
                    ocr_system = (
                        "You are an OCR reader for manga speech bubbles. "
                        "Read the text inside each numbered green box. "
                        "Identify the speaker from visual cues (bubble tail direction, character position). "
                        "Format: NUMBER. [speaker] text. "
                        "Speakers: girl, boy, man, woman, child, narrator, unknown. "
                        "No translations. No descriptions. No reasoning. No thinking. "
                        "Start your first line with '1.'"
                    )

                resp = requests.post(
                    FIREWORKS_URL,
                    headers={
                        "Authorization": f"Bearer {_get_api_key()}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": VISION_MODEL,
                        "max_tokens": 4096,
                        "temperature": 0.1,
                        "thinking": {"type": "disabled"},
                        "messages": [
                            {
                                "role": "system",
                                "content": ocr_system,
                            },
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image_url",
                                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                                ],
                            },
                        ],
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                msg = data["choices"][0]["message"]
                content = msg.get("content") or ""
                content = content.strip()
                if not content:
                    finish = data["choices"][0].get("finish_reason", "?")
                    print(f"[OCR] Attempt {attempt+1}: empty response, finish_reason={finish}")
                    if attempt < 2:
                        continue
                print(f"[OCR] Vision raw response:\n{content}")
                return self._parse_response(content, bboxes, num)
            except Exception as e:
                print(f"[OCR] Vision API error (attempt {attempt+1}): {e}")
                if attempt == 2:
                    return []
        return []

    def _parse_response(self, content: str, bboxes, expected: int) -> List[TextRegion]:
        results = {}
        current_num = None
        current_text = ""

        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue

            match = re.match(r'^(\d+)\s*[.)\]:]\s*(.*)', line)
            if match:
                if current_num is not None and current_text.strip():
                    results[current_num] = current_text.strip()
                current_num = int(match.group(1))
                current_text = match.group(2).strip()
            else:
                if current_num is not None:
                    current_text += " " + line

        if current_num is not None and current_text.strip():
            results[current_num] = current_text.strip()

        for num, text in list(results.items()):
            if not (1 <= num <= expected):
                del results[num]
                continue
            speaker = ""
            sp_match = re.match(r'\[(\w+)\]\s*(.*)', text)
            if sp_match:
                speaker = sp_match.group(1)
                text = sp_match.group(2).strip()
            text = text.strip('"\'""''')
            if text.lower().startswith('box') and ':' in text:
                colon_pos = text.index(':')
                after_colon = text[colon_pos + 1:].strip().strip('"\'""''')
                if after_colon:
                    text = after_colon
            text = text.strip()
            if text.lower() in ('[unreadable]', 'unreadable', 'n/a', '-'):
                del results[num]
                continue
            results[num] = {"text": text, "speaker": speaker}

        out = []
        for i, (bx, by, bw, bh) in enumerate(bboxes):
            n = i + 1
            entry = results.get(n)
            if entry:
                out.append(((bx, by, bw, bh), entry["text"]))
                sp = entry["speaker"]
                if sp and sp != "unknown":
                    print(f"  #{n} ({bx},{by},{bw}x{bh}) [{sp}]: '{entry['text']}'")
                else:
                    print(f"  #{n} ({bx},{by},{bw}x{bh}): '{entry['text']}'")
            else:
                print(f"  #{n} ({bx},{by},{bw}x{bh}): not read")
        return out
