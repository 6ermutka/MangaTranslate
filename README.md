# MangaTranslate

Веб-сервис для автоматического перевода манги с английского на русский.  
Скачивает главы с MangaDex, распознаёт текст в пузырях (YOLO + Qwen3 Vision), переводит (Llama 3.3 70B) и собирает PDF.

![Python](https://img.shields.io/badge/Python-3.11+-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)

---

## Возможности

- Парсинг манги по ссылке или ID с MangaDex
- Детектирование речевых пузырей (кастомная YOLO модель)
- OCR с определением говорящего (Qwen3 Vision через Fireworks AI)
- Перевод с учётом русской грамматики, падежей и порядка слов (Llama 3.3 70B)
- Inpainting фона под пузырями, рендер переведённого текста
- Экспорт в PDF
- Пауза / возобновление перевода, автоматическое продолжение после перезапуска

---

## Требования

- Python 3.11+
- Аккаунт на [Fireworks AI](https://fireworks.ai) (бесплатный tier достаточен)

---

## Установка

### 1. Клонировать репозиторий

```bash
git clone https://github.com/YOUR_USERNAME/MangaTranslate.git
cd MangaTranslate
```

### 2. Установить зависимости

```bash
pip install -r requirements.txt
```

### 3. Указать API ключ

Открой `config/web_config.json` и замени `YOUR_FIREWORKS_API_KEY` на свой ключ с [fireworks.ai](https://fireworks.ai/account/api-keys):

```json
{
    "fireworks_api_key": "fw_xxxxxxxxxxxxxxxxxxxxxxxx",
    ...
}
```

> Ключ также можно передать через переменную окружения: `export FIREWORKS_API_KEY=fw_xxx`

### 4. Запустить сервер

```bash
python3.11 web/server.py
```

Открой в браузере: **http://localhost:8420**

---

## Использование

1. **Парсинг** — вставь ссылку с MangaDex (например `https://mangadex.org/title/...`) или ID тайтла
2. **Выбери главы** для перевода
3. **Нажми «Перевести»** — главы встанут в очередь и начнут обрабатываться в фоне
4. **Мои тайтлы** — следи за прогрессом, ставь на паузу ⏸ / возобновляй ▶, открывай готовые PDF

---

## Структура проекта

```
MangaTranslate/
├── web/
│   ├── server.py          # FastAPI сервер
│   └── static/index.html  # Веб-интерфейс
├── ocr/
│   ├── ocr_engine.py      # OCR через Qwen3 Vision
│   └── yolo_detector.py   # Детектор пузырей
├── translation/
│   └── translator.py      # Перевод через Llama 3.3 70B
├── processing/
│   ├── pipeline.py        # Inpainting фона
│   └── text_overlay.py    # Рендер текста
├── models/
│   └── bubble_detector.pt # Обученная YOLO модель
├── fonts/                 # Шрифты для рендера
├── config/
│   └── web_config.json    # Настройки (API ключ, промпты, словарь)
└── requirements.txt
```

---

## Настройки (`config/web_config.json`)

| Параметр | Описание |
|---|---|
| `fireworks_api_key` | API ключ Fireworks AI |
| `fireworks_vision_model` | Модель OCR (по умолчанию Qwen3) |
| `fireworks_translate_model` | Модель перевода (по умолчанию Llama 3.3 70B) |
| `default_source_lang` | Язык оригинала (`en`, `ja`, `id`) |
| `ocr_system_prompt` | Промпт для OCR модели |
| `system_prompt` | Промпт для модели перевода |
| `user_dict` | Словарь замен (например суффиксы `-kun`, `-chan`) |
| `port` | Порт сервера (по умолчанию 8420) |

Все параметры также редактируются через вкладку **Настройки** в веб-интерфейсе.
