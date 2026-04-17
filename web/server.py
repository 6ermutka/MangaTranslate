from __future__ import annotations

import io
import json
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "config" / "web_config.json"
DATA_DIR = PROJECT_ROOT / "web" / "data"
STATIC_DIR = PROJECT_ROOT / "web" / "static"

API_BASE = "https://api.mangadex.org"

app = FastAPI()

_stop_flags: dict[str, threading.Event] = {}


def init_dirs():
    for d in ["covers", "pdf", "pages"]:
        (DATA_DIR / d).mkdir(parents=True, exist_ok=True)


init_dirs()

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/covers", StaticFiles(directory=str(DATA_DIR / "covers")), name="covers")
app.mount("/pdf", StaticFiles(directory=str(DATA_DIR / "pdf")), name="pdf")

_lib_lock = threading.Lock()


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def save_config(cfg: dict):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)


def load_library() -> dict:
    with _lib_lock:
        lib_path = DATA_DIR / "library.json"
        if lib_path.exists():
            with open(lib_path) as f:
                return json.load(f)
        return {"titles": {}}


def save_library(lib: dict):
    with _lib_lock:
        lib_path = DATA_DIR / "library.json"
        with open(lib_path, "w") as f:
            json.dump(lib, f, indent=2, ensure_ascii=False)


def update_chapter(manga_id: str, ch_num: str, updates: dict):
    with _lib_lock:
        lib_path = DATA_DIR / "library.json"
        if lib_path.exists():
            with open(lib_path) as f:
                lib = json.load(f)
        else:
            lib = {"titles": {}}
        if manga_id in lib["titles"] and ch_num in lib["titles"][manga_id]["chapters"]:
            for k, v in updates.items():
                if v is None:
                    lib["titles"][manga_id]["chapters"][ch_num].pop(k, None)
                else:
                    lib["titles"][manga_id]["chapters"][ch_num][k] = v
            with open(lib_path, "w") as f:
                json.dump(lib, f, indent=2, ensure_ascii=False)


def fetch_manga_info(manga_id: str) -> dict:
    resp = requests.get(f"{API_BASE}/manga/{manga_id}", timeout=30)
    resp.raise_for_status()
    data = resp.json()["data"]
    attrs = data["attributes"]

    title = ""
    for lang in ["en", "ja-ro", "ja", "id"]:
        if lang in attrs.get("title", {}):
            title = attrs["title"][lang]
            break
    if not title:
        title = list(attrs.get("title", {}).values())[0] if attrs.get("title") else manga_id

    cover_id = None
    for rel in data.get("relationships", []):
        if rel["type"] == "cover_art":
            cover_id = rel["id"]
            break

    cover_url = None
    if cover_id:
        try:
            cr = requests.get(f"{API_BASE}/cover/{cover_id}", timeout=15)
            cr.raise_for_status()
            fname = cr.json()["data"]["attributes"]["fileName"]
            cover_url = f"https://uploads.mangadex.org/covers/{manga_id}/{fname}"
        except Exception:
            pass

    return {"id": manga_id, "title": title, "cover_url": cover_url}


def fetch_chapters_list(manga_id: str, lang: str = "en") -> list:
    chapters = []
    offset = 0
    while True:
        resp = requests.get(
            f"{API_BASE}/manga/{manga_id}/feed",
            params={"translatedLanguage[]": lang, "order[chapter]": "asc", "limit": 100, "offset": offset},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        for ch in data["data"]:
            attrs = ch["attributes"]
            chapters.append({
                "id": ch["id"],
                "chapter": attrs.get("chapter") or "?",
                "title": attrs.get("title") or "",
                "pages": attrs.get("pages") or 0,
                "volume": attrs.get("volume") or "",
                "lang": attrs.get("translatedLanguage", lang),
            })
        offset += 100
        if offset >= data["total"]:
            break
    return chapters


def download_cover(manga_id: str, cover_url: str) -> str:
    covers_dir = DATA_DIR / "covers"
    covers_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{manga_id}.jpg"
    out = covers_dir / fname
    if out.exists():
        return fname
    try:
        resp = requests.get(cover_url, timeout=30)
        resp.raise_for_status()
        out.write_bytes(resp.content)
        return fname
    except Exception:
        return ""


def translate_chapter_background(manga_id: str, chapter_id: str, ch_num: str, source_lang: str,
                                  stop_event: threading.Event = None):
    if stop_event is None:
        stop_event = threading.Event()
    flag_key = f"{manga_id}:{ch_num}"
    _stop_flags[flag_key] = stop_event

    tid = manga_id
    ch_key = ch_num

    lib = load_library()
    if tid not in lib["titles"]:
        return
    if ch_key in lib["titles"][tid].get("chapters", {}):
        if lib["titles"][tid]["chapters"][ch_key].get("status") == "done":
            return

    from ocr.ocr_engine import OCREngine
    from translation.translator import Translator
    from processing.pipeline import process_contour, render_text, CHAR_FIXES
    from fpdf import FPDF
    from PIL import Image as PILImage
    import cv2
    import numpy as np

    ocr = OCREngine()
    translator = Translator()

    pdf_dir = DATA_DIR / "pdf" / tid
    pages_dir = DATA_DIR / "pages" / tid / ch_key
    pages_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    already_done = len(list(pages_dir.glob("*.png")))
    if already_done > 0:
        update_chapter(tid, ch_key, {"status": "translating", "pages_done": already_done})
    else:
        update_chapter(tid, ch_key, {"status": "translating", "progress": 0, "pages_done": 0, "pages_total": 0})

    try:
        resp = requests.get(f"{API_BASE}/at-home/server/{chapter_id}", timeout=30)
        resp.raise_for_status()
        data = resp.json()
        base_url = data["baseUrl"]
        h = data["chapter"]["hash"]
        page_urls = [f"{base_url}/data/{h}/{p}" for p in data["chapter"]["data"]]
    except Exception as e:
        update_chapter(tid, ch_key, {"status": "error", "error": str(e)})
        return

    total = len(page_urls)
    update_chapter(tid, ch_key, {"pages_total": total})

    for i, url in enumerate(page_urls):
        if stop_event.is_set():
            update_chapter(tid, ch_key, {"status": "paused"})
            return

        page_path = pages_dir / f"{i+1:03d}.png"
        if page_path.exists():
            update_chapter(tid, ch_key, {"progress": int((i + 1) / total * 100), "pages_done": i + 1})
            continue

        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            pil_img = PILImage.open(io.BytesIO(resp.content)).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            if img is None:
                continue
        except Exception:
            continue

        img_h, img_w = img.shape[:2]
        ocr_results = ocr.detect(img, source_lang)

        canvas_bgr = img.copy()
        regions_to_translate = []
        box_data = []

        for bbox, orig in ocr_results:
            bx, by, bw, bh = bbox
            x1, y1 = max(0, bx), max(0, by)
            x2, y2 = min(img_w, bx + bw), min(img_h, by + bh)
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            cleaned, contour = process_contour(crop)
            canvas_bgr[y1:y2, x1:x2] = cleaned
            is_dark = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean() < 128
            if orig:
                regions_to_translate.append((bbox, orig))
                box_data.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                 'contour': contour, 'is_dark': is_dark})

        if regions_to_translate:
            translated_regions = translator.translate_regions(
                regions_to_translate, source_lang=source_lang, target_lang="ru")
            translated_map = {bbox: tr for bbox, _, tr in translated_regions}
            canvas_pil = PILImage.fromarray(cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB))
            for bd in box_data:
                bbox = (bd['x1'], bd['y1'], bd['x2'] - bd['x1'], bd['y2'] - bd['y1'])
                translated = translated_map.get(bbox, "")
                if translated:
                    for old, new in CHAR_FIXES.items():
                        translated = translated.replace(old, new)
                    render_text(canvas_pil, bd['x1'], bd['y1'], bd['x2'], bd['y2'],
                                translated, contour=bd['contour'], is_dark=bd['is_dark'])
            result = np.array(canvas_pil)
        else:
            result = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)

        page_path = pages_dir / f"{i+1:03d}.png"
        PILImage.fromarray(result).save(str(page_path))

        update_chapter(tid, ch_key, {"progress": int((i + 1) / total * 100), "pages_done": i + 1})

    DPI = 150
    MM_PER_INCH = 25.4
    pages = sorted(pages_dir.glob("*.png"))
    if pages:
        pdf = FPDF(orientation="P", unit="mm", format="A4")
        pdf.set_auto_page_break(False)
        for p in pages:
            im = PILImage.open(str(p))
            pw, ph = im.size
            im.close()
            w_mm = pw / DPI * MM_PER_INCH
            h_mm = ph / DPI * MM_PER_INCH
            orientation = "L" if pw > ph else "P"
            pdf.add_page(orientation=orientation, format=(w_mm, h_mm))
            pdf.image(str(p), x=0, y=0, w=w_mm, h=h_mm)
        pdf.output(str(pdf_dir / f"chapter_{ch_key}.pdf"))

    shutil.rmtree(pages_dir, ignore_errors=True)

    final = {"status": "done", "progress": 100, "pdf": f"chapter_{ch_key}.pdf"}
    lib = load_library()
    if "error" in lib["titles"][tid]["chapters"][ch_key]:
        final["error"] = None
    update_chapter(tid, ch_key, final)


# ── Routes ────────────────────────────────────────────────────


@app.on_event("startup")
async def resume_interrupted():
    lib = load_library()
    for tid, title in lib["titles"].items():
        for ch_num, ch in title["chapters"].items():
            if ch.get("status") in ("translating", "queued", "paused"):
                stop_event = threading.Event()
                t = threading.Thread(
                    target=translate_chapter_background,
                    args=(tid, ch["id"], ch_num, ch.get("lang", title.get("source_lang", "en")), stop_event),
                    daemon=True,
                )
                t.start()


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/config")
async def get_config():
    return load_config()


@app.post("/api/config")
async def update_config(body: dict):
    cfg = load_config()
    for k, v in body.items():
        if k in cfg:
            cfg[k] = v
    save_config(cfg)
    return cfg


@app.get("/api/library")
async def get_library():
    return load_library()


class ParseRequest(BaseModel):
    url: str
    lang: str = "en"


@app.post("/api/parse")
async def parse_manga(req: ParseRequest):
    url = req.url
    if "/" in url:
        parts = url.strip("/").split("/")
        for i, p in enumerate(parts):
            if p == "title" and i + 1 < len(parts):
                manga_id = parts[i + 1]
                break
        else:
            raise HTTPException(400, "Cannot extract manga ID")
    else:
        manga_id = url

    info = fetch_manga_info(manga_id)
    chapters = fetch_chapters_list(manga_id, req.lang)

    return {"manga": info, "chapters": chapters}


class TranslateRequest(BaseModel):
    manga_id: str
    manga_title: str
    cover_url: Optional[str] = None
    chapters: list
    source_lang: str = "en"


@app.post("/api/translate")
async def start_translation(req: TranslateRequest):
    lib = load_library()
    tid = req.manga_id

    if tid not in lib["titles"]:
        cover_fname = ""
        if req.cover_url:
            cover_fname = download_cover(tid, req.cover_url)
        lib["titles"][tid] = {
            "id": tid,
            "title": req.manga_title,
            "cover": cover_fname,
            "source_lang": req.source_lang,
            "chapters": {},
        }

    for ch in req.chapters:
        ch_num = ch["chapter"]
        ch_id = ch["id"]
        if ch_num in lib["titles"][tid]["chapters"]:
            if lib["titles"][tid]["chapters"][ch_num].get("status") == "done":
                continue
        lib["titles"][tid]["chapters"][ch_num] = {
            "id": ch_id,
            "chapter": ch_num,
            "title": ch.get("title", ""),
            "pages": ch.get("pages", 0),
            "lang": ch.get("lang", req.source_lang),
            "status": "queued",
            "progress": 0,
            "pages_done": 0,
            "pages_total": 0,
        }

    save_library(lib)

    for ch in req.chapters:
        ch_num = ch["chapter"]
        ch_id = ch["id"]
        lang = ch.get("lang", req.source_lang)
        t = threading.Thread(
            target=translate_chapter_background,
            args=(tid, ch_id, ch_num, lang),
            daemon=True,
        )
        t.start()

    return {"status": "started"}


@app.post("/api/pause/{manga_id}/{ch_num}")
async def pause_chapter(manga_id: str, ch_num: str):
    flag_key = f"{manga_id}:{ch_num}"
    if flag_key in _stop_flags:
        _stop_flags[flag_key].set()
    update_chapter(manga_id, ch_num, {"status": "paused"})
    return {"paused": True}


@app.post("/api/resume/{manga_id}/{ch_num}")
async def resume_chapter(manga_id: str, ch_num: str):
    lib = load_library()
    if manga_id not in lib["titles"] or ch_num not in lib["titles"][manga_id]["chapters"]:
        raise HTTPException(404, "Chapter not found")
    ch = lib["titles"][manga_id]["chapters"][ch_num]
    update_chapter(manga_id, ch_num, {"status": "translating"})
    stop_event = threading.Event()
    t = threading.Thread(
        target=translate_chapter_background,
        args=(manga_id, ch["id"], ch_num, ch.get("lang", lib["titles"][manga_id].get("source_lang", "en")), stop_event),
        daemon=True,
    )
    t.start()
    return {"resumed": True}


@app.delete("/api/title/{manga_id}")
async def delete_title(manga_id: str):
    lib = load_library()
    if manga_id in lib["titles"]:
        del lib["titles"][manga_id]
        save_library(lib)
    pdf_dir = DATA_DIR / "pdf" / manga_id
    if pdf_dir.exists():
        shutil.rmtree(pdf_dir)
    cover = DATA_DIR / "covers" / f"{manga_id}.jpg"
    if cover.exists():
        cover.unlink()
    return {"deleted": True}


@app.get("/api/view/{manga_id}/{ch_num}")
async def view_pdf(manga_id: str, ch_num: str):
    pdf_path = DATA_DIR / "pdf" / manga_id / f"chapter_{ch_num}.pdf"
    if not pdf_path.exists():
        raise HTTPException(404, "PDF not found")
    return FileResponse(str(pdf_path), media_type="application/pdf")


@app.delete("/api/chapter/{manga_id}/{ch_num}")
async def delete_chapter(manga_id: str, ch_num: str):
    lib = load_library()
    if manga_id in lib["titles"] and ch_num in lib["titles"][manga_id]["chapters"]:
        del lib["titles"][manga_id]["chapters"][ch_num]
        save_library(lib)
    pdf_path = DATA_DIR / "pdf" / manga_id / f"chapter_{ch_num}.pdf"
    if pdf_path.exists():
        pdf_path.unlink()
    return {"deleted": True}


if __name__ == "__main__":
    import uvicorn
    cfg = load_config()
    port = cfg.get("port", 8420)
    uvicorn.run(app, host="0.0.0.0", port=port)
