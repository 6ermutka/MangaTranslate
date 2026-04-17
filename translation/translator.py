from __future__ import annotations

import json
import requests
from collections import OrderedDict
from typing import List, Tuple

TextRegion = Tuple[Tuple[int, int, int, int], str]
TranslatedRegion = Tuple[Tuple[int, int, int, int], str, str]

FIREWORKS_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
FIREWORKS_MODEL = "accounts/fireworks/models/llama-v3p3-70b-instruct"


def _get_api_key() -> str:
    try:
        from pathlib import Path
        import json as _j
        cfg = _j.load(open(Path(__file__).parent.parent / "config" / "web_config.json"))
        key = cfg.get("fireworks_api_key", "")
        if key and not key.startswith("YOUR_"):
            return key
    except Exception:
        pass
    import os
    return os.getenv("FIREWORKS_API_KEY", "")

SYSTEM_PROMPT_DEFAULT = (
    "Ты — профессиональный переводчик манги с английского на русский.\n\n"
    "ГРАММАТИКА РУССКОГО ЯЗЫКА:\n"
    "• Падежи: согласовывай слова по падежу (И/Р/Д/В/Т/П). "
    "Пример: «идти к другу» (Д.п.), «думать о школе» (П.п.), «взять книгу» (В.п.).\n"
    "• Порядок слов: в русском он свободный, но логичный. "
    "Новая информация — в конец фразы, тема — в начало. "
    "Избегай дословного калькирования английского порядка SVO там, где это звучит неестественно.\n"
    "• Согласование: прилагательные, причастия и местоимения согласуй с существительным по роду, числу и падежу.\n"
    "• Глаголы: выбирай вид (совершенный/несовершенный) по смыслу фразы.\n\n"
    "ИМЕНА СОБСТВЕННЫЕ:\n"
    "• НЕ переводи имена персонажей — оставляй как в оригинале (например: Naruto, Sakura, Lena, John).\n"
    "• Имена можно склонять по падежам, если это нужно по-русски "
    "(например: «Привет, Naruto!» → «Привет, Naruto!»; «письмо от Sakura» → «письмо от Sakura»).\n"
    "• Названия мест, техник, специальных терминов манги — оставляй в оригинале или транслитерируй, не переводи.\n\n"
    "СТИЛЬ И ФОРМАТИРОВАНИЕ:\n"
    "1. Переводи смысл, а не пословно. Русский должен звучать естественно, как живой разговор.\n"
    "2. Исходный текст часто написан КАПСОМ — это стиль манги, а не крик. Переводи обычным регистром.\n"
    "3. Сохраняй эмоции и пунктуацию: ... !? — ♡ ~\n"
    "4. Если указан говорящий ([girl], [boy] и т.д.), учитывай пол: girl=она/её, boy=он/его.\n"
    "5. Выводи ТОЛЬКО переведённые строки, столько же сколько исходных. Без пояснений."
)

RU_FIXES = {
    'дата': 'свидание', 'Дата': 'Свидание', 'ДАТА': 'СВИДАНИЕ',
    'перерыв': 'каникулы', 'Перерыв': 'Каникулы', 'ПЕРЕРЫВ': 'КАНИКУЛЫ',
    'девочка': 'девушка', 'Девочка': 'Девушка', 'ДЕВОЧКА': 'ДЕВУШКА',
}


def _build_system_prompt() -> str:
    prompt = SYSTEM_PROMPT_DEFAULT
    try:
        from pathlib import Path
        cfg_path = Path(__file__).parent.parent / "config" / "web_config.json"
        if cfg_path.exists():
            import json as _json
            with open(cfg_path) as _f:
                cfg = _json.load(_f)
            if cfg.get("system_prompt"):
                prompt = cfg["system_prompt"]
            if cfg.get("user_dict"):
                dict_lines = [f"{k} → {v}" for k, v in cfg["user_dict"].items()]
                prompt += "\n\n" + "Переводи согласно словарю:\n" + "\n".join(dict_lines)
            return prompt
    except Exception:
        pass
    from config.settings import settings
    if settings.user_dict:
        dict_lines = [f"{k} → {v}" for k, v in settings.user_dict.items()]
        prompt += "\n\n6. Переводи согласно словарю:\n" + "\n".join(dict_lines)
    return prompt


def _is_garbage(text: str) -> bool:
    alpha = sum(1 for c in text if c.isalpha())
    if alpha < 3:
        return True
    words = text.split()
    if len(words) < 2:
        return False
    short = sum(1 for w in words if len(w) <= 2 and w.isalpha())
    if short / len(words) > 0.6 and len(words) > 5:
        return True
    gibberish = sum(1 for w in words if not any(c in 'aeiouAEIOU' for c in w) and len(w) > 3 and w.isalpha())
    if gibberish > len(words) * 0.4:
        return True
    return False


def _fix_ru(text: str) -> str:
    for wrong, right in RU_FIXES.items():
        text = text.replace(wrong, right)
    return text


class Translator:
    def __init__(self):
        self._text_cache = OrderedDict()
        self._max_cache_size = 1000

    def translate_regions(
        self,
        regions: List[TextRegion],
        source_lang: str = "auto",
        target_lang: str = "ru",
    ) -> List[TranslatedRegion]:
        if not regions:
            return []

        texts = []
        valid_indices = []
        for i, (bbox, text) in enumerate(regions):
            if _is_garbage(text):
                continue
            texts.append(text)
            valid_indices.append(i)

        if not texts:
            return [(bbox, text, text) for bbox, text in regions]

        from config.settings import settings

        translations = self._batch_translate_fireworks(texts, source_lang, target_lang)

        results = []
        ti = 0
        for i, (bbox, text) in enumerate(regions):
            if i in valid_indices:
                translated = translations[ti] if ti < len(translations) else text
                translated = _fix_ru(translated)
                ti += 1
            else:
                translated = text
            print(f"[Translate] '{text}' → '{translated}'")
            results.append((bbox, text, translated))
        return results

    def _batch_translate_fireworks(self, texts: List[str], source: str, target: str) -> List[str]:
        if len(texts) == 1:
            return [self._fireworks_single(texts[0], source, target)]
        src_name = {"en": "English", "ja": "Japanese", "auto": "auto-detect"}.get(source, source)
        tgt_name = {"ru": "Russian", "en": "English"}.get(target, target)
        numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
        prompt = f"Translate from {src_name} to {tgt_name}. Output in {tgt_name.upper()}:\n\n{numbered}"
        try:
            resp = requests.post(
                FIREWORKS_URL,
                headers={
                    "Authorization": f"Bearer {_get_api_key()}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": FIREWORKS_MODEL,
                    "max_tokens": 4096,
                    "temperature": 0.35,
                    "top_p": 0.85,
                    "presence_penalty": 0.1,
                    "frequency_penalty": 0.1,
                    "messages": [
                        {"role": "system", "content": _build_system_prompt()},
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout=20,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            lines = content.split("\n")
            results = []
            for line in lines:
                cleaned = line.strip()
                for prefix in [f"{len(results)+1}.", f"{len(results)+1})", f"{len(results)+1}:", "•", "-"]:
                    if cleaned.startswith(prefix):
                        cleaned = cleaned[len(prefix):].strip()
                        break
                if cleaned:
                    results.append(cleaned)
            while len(results) < len(texts):
                results.append(texts[len(results)])
            return results[:len(texts)]
        except Exception as e:
            print(f"[Translate] Fireworks error: {e}")
            return texts[:]

    def _fireworks_single(self, text: str, source: str, target: str) -> str:
        cache_key = ("fw", source, target, hash(text))
        if cache_key in self._text_cache:
            self._text_cache.move_to_end(cache_key)
            return self._text_cache[cache_key]

        src_name = {"en": "English", "ja": "Japanese", "auto": "auto-detect"}.get(source, source)
        tgt_name = {"ru": "Russian", "en": "English"}.get(target, target)
        prompt = f"Translate from {src_name} to {tgt_name}. Output in {tgt_name.upper()}:\n\n{text}"
        try:
            resp = requests.post(
                FIREWORKS_URL,
                headers={
                    "Authorization": f"Bearer {_get_api_key()}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": FIREWORKS_MODEL,
                    "max_tokens": 1024,
                    "temperature": 0.35,
                    "top_p": 0.85,
                    "presence_penalty": 0.1,
                    "frequency_penalty": 0.1,
                    "messages": [
                        {"role": "system", "content": _build_system_prompt()},
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout=10,
            )
            resp.raise_for_status()
            result = resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[Translate] Fireworks error: {e}")
            result = text

        self._text_cache[cache_key] = result
        return result
