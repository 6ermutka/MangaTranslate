from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path
from typing import Callable, Optional, Dict, List
from dataclasses import dataclass


HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "translation"


@dataclass
class ModelInfo:
    id: str
    name: str
    size_mb: int
    source_lang: str
    target_lang: str


AVAILABLE_MODELS: Dict[str, ModelInfo] = {
    "en-ru": ModelInfo(
        id="Helsinki-NLP/opus-mt-en-ru",
        name="Английский → Русский",
        size_mb=300,
        source_lang="en",
        target_lang="ru",
    ),
    "ja-en": ModelInfo(
        id="Helsinki-NLP/opus-mt-ja-en",
        name="Японский → Английский",
        size_mb=307,
        source_lang="ja",
        target_lang="en",
    ),
    "zh-en": ModelInfo(
        id="Helsinki-NLP/opus-mt-zh-en",
        name="Китайский → Английский",
        size_mb=300,
        source_lang="zh-CN",
        target_lang="en",
    ),
    "ko-en": ModelInfo(
        id="Helsinki-NLP/opus-mt-ko-en",
        name="Корейский → Английский",
        size_mb=300,
        source_lang="ko",
        target_lang="en",
    ),
}

LANG_TO_INTERMEDIATE = {
    "ja": "ja-en",
    "zh-CN": "zh-en",
    "ko": "ko-en",
}


def get_required_models(source_lang: str, target_lang: str) -> List[str]:
    models = []
    if target_lang == "ru" and source_lang not in ("en", "auto"):
        intermediate = LANG_TO_INTERMEDIATE.get(source_lang)
        if intermediate:
            models.append(intermediate)
    if target_lang == "ru":
        models.append("en-ru")
    return models


def is_downloaded(model_key: str) -> bool:
    model = AVAILABLE_MODELS.get(model_key)
    if not model:
        return False
    hf_dir = HF_CACHE / f"models--{model.id.replace('/', '--')}"
    if hf_dir.exists():
        return True
    return False


def get_download_size_mb(model_key: str) -> int:
    model = AVAILABLE_MODELS.get(model_key)
    return model.size_mb if model else 0


def delete_model(model_key: str) -> bool:
    model = AVAILABLE_MODELS.get(model_key)
    if not model:
        return False
    hf_dir = HF_CACHE / f"models--{model.id.replace('/', '--')}"
    if hf_dir.exists():
        shutil.rmtree(hf_dir)
        print(f"[ModelManager] Удалена модель: {model.name}")
        return True
    return False


def download_model(
    model_key: str,
    on_progress: Optional[Callable[[str, int], None]] = None,
    on_done: Optional[Callable[[bool], None]] = None,
) -> threading.Thread:
    def _download():
        model = AVAILABLE_MODELS.get(model_key)
        if not model:
            if on_done:
                on_done(False)
            return

        try:
            if on_progress:
                on_progress(f"Скачивание: {model.name}...", 0)

            from transformers import MarianMTModel, MarianTokenizer

            name = model.id
            if on_progress:
                on_progress(f"Загрузка токенизатора {model.name}...", 20)
            MarianTokenizer.from_pretrained(name)

            if on_progress:
                on_progress(f"Загрузка модели {model.name}...", 50)
            MarianMTModel.from_pretrained(name)

            if on_progress:
                on_progress(f"{model.name} готова!", 100)

            print(f"[ModelManager] Скачана модель: {model.name}")
            if on_done:
                on_done(True)

        except Exception as e:
            print(f"[ModelManager] Ошибка скачивания: {e}")
            if on_progress:
                on_progress(f"Ошибка: {e}", -1)
            if on_done:
                on_done(False)

    t = threading.Thread(target=_download, daemon=True)
    t.start()
    return t


def get_downloaded_models() -> List[str]:
    return [k for k in AVAILABLE_MODELS if is_downloaded(k)]


def all_required_downloaded(source_lang: str, target_lang: str) -> bool:
    required = get_required_models(source_lang, target_lang)
    return all(is_downloaded(k) for k in required)
