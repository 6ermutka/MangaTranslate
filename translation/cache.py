from collections import OrderedDict
from typing import Optional, List, Tuple
from config.settings import settings


CacheEntry = List[Tuple[Tuple[int, int, int, int], str, str]]
# [(bbox, original_text, translated_text), ...]


class TranslationCache:
    """LRU кэш: хэш изображения → список переведённых регионов."""

    def __init__(self, max_size: int = None):
        self._max_size = max_size or settings.cache_max_size
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()

    def get(self, image_hash: str) -> Optional[CacheEntry]:
        if image_hash not in self._cache:
            return None
        # LRU: переносим в конец как recently used
        self._cache.move_to_end(image_hash)
        return self._cache[image_hash]

    def set(self, image_hash: str, entry: CacheEntry):
        if image_hash in self._cache:
            self._cache.move_to_end(image_hash)
        self._cache[image_hash] = entry
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)  # удаляем самый старый

    def clear(self):
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)
