"""
FastAPI app — upload a batch of manga pages, confirm/reorder them, pick a
target language, translate, view/download the result.

Deliberately plain: server-rendered templates + a little vanilla JS, SQLite,
one background worker. See README for why (this is meant to run on the
cheapest server you've got, for one person).
"""
from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageDraw

from . import db
from .config import settings
from .pipeline.ordering import natural_sort_key
from .pipeline.render import layout_line, render_page
from .pipeline.worker import enqueue_batch

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Manga Translator")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def _static_url(name: str) -> str:
    """A static asset URL with a `?v=<mtime>` cache-buster.

    Without this, a browser that already has a page open (or that just
    caches aggressively) can keep running an old cached copy of app.js
    indefinitely after a real code change — which looks exactly like "the
    button doesn't do anything", since the JS actually handling that click
    was never fetched. Tying the query string to the file's own last-
    modified time means any deploy that changes the file gets a new URL
    automatically, with nothing to remember to bump by hand.
    """
    path = BASE_DIR / "static" / name
    try:
        version = int(path.stat().st_mtime)
    except OSError:
        version = 0
    return f"/static/{name}?v={version}"


templates.env.globals["static_url"] = _static_url


def _rgb_json_to_hex(rgb_json: str) -> str:
    r, g, b = json.loads(rgb_json)
    return f"#{r:02x}{g:02x}{b:02x}"


templates.env.filters["rgb_hex"] = _rgb_json_to_hex
templates.env.filters["to_json"] = json.dumps


@app.on_event("startup")
def _startup() -> None:
    db.init_db()


# ---------------------------------------------------------------------------
# Home — upload form + batch list
# ---------------------------------------------------------------------------

@app.get("/")
def index(request: Request):
    batches = db.list_batches()
    return templates.TemplateResponse(request, "index.html", {"batches": batches})


@app.post("/batches")
async def create_batch(
    files: list[UploadFile] = File(...),
    target_lang: str = Form("English"),
):
    if not files:
        raise HTTPException(400, "No files uploaded")

    batch_id = db.create_batch(target_lang=target_lang)
    batch_dir = settings.uploads_dir / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Propose an order via natural sort of the original filenames (so
    # "2.jpg" sorts before "10.jpg") — the user confirms/fixes it next.
    ordered = sorted(files, key=lambda f: natural_sort_key(f.filename or ""))
    for index_, upload in enumerate(ordered):
        dest = batch_dir / (upload.filename or f"page_{index_}")
        with dest.open("wb") as out:
            shutil.copyfileobj(upload.file, out)
        with Image.open(dest) as img:
            width, height = img.size
        db.add_page(batch_id, index_, upload.filename or dest.name, str(dest), width, height)

    return RedirectResponse(f"/batches/{batch_id}", status_code=303)


# ---------------------------------------------------------------------------
# Batch detail — reorder, start translation, view results
# ---------------------------------------------------------------------------

def _current_boxes(page: sqlite3.Row, lines: list[sqlite3.Row]) -> dict[str, dict]:
    """For the correction UI: what x/y/w/h/font size each line is
    *actually* rendered with right now — the automatic layout, or a
    manual override if one is set — so those fields can be pre-filled
    with real numbers instead of a blank field that just says "auto".
    Untranslated lines have nothing rendered yet, so they fall back to
    their raw detected box with no font size.
    """
    all_boxes = [(r["id"], (r["x"], r["y"], r["w"], r["h"])) for r in lines]
    current: dict[str, dict] = {}
    img = None
    draw = None
    for r in lines:
        if not (r["translated_text"] or "").strip():
            current[r["id"]] = {"x": r["x"], "y": r["y"], "w": r["w"], "h": r["h"], "font_size": None}
            continue
        if img is None:
            img = Image.open(page["orig_path"]).convert("RGB")
            draw = ImageDraw.Draw(img)
        others = [box for line_id, box in all_boxes if line_id != r["id"]]
        (x, y, w, h), font, _ = layout_line(draw, r, others, page["width"], page["height"])
        current[r["id"]] = {
            "x": round(x), "y": round(y), "w": round(w), "h": round(h),
            "font_size": getattr(font, "size", None),
        }
    return current


@app.get("/batches/{batch_id}")
def batch_detail(request: Request, batch_id: str):
    batch = db.get_batch(batch_id)
    if batch is None:
        raise HTTPException(404, "Batch not found")
    pages = db.list_pages(batch_id)
    pages_with_lines = [(page, db.list_lines(page["id"])) for page in pages]
    current_boxes: dict[str, dict] = {}
    for page, lines in pages_with_lines:
        current_boxes.update(_current_boxes(page, lines))
    return templates.TemplateResponse(
        request,
        "batch.html",
        {
            "batch": batch,
            "pages_with_lines": pages_with_lines,
            "page_ids": [page["id"] for page in pages],  # in reading order — see the page reader in app.js
            "current_boxes": current_boxes,  # line_id -> {x, y, w, h, font_size} actually in effect right now
        },
    )


@app.post("/batches/{batch_id}/move")
async def move_page(batch_id: str, request: Request):
    """Move one page up/down in reading order. Body: {"page_id": ..., "direction": "up"|"down"}."""
    body = await request.json()
    page_id, direction = body.get("page_id"), body.get("direction")
    pages = db.list_pages(batch_id)
    ids = [p["id"] for p in pages]
    if page_id not in ids:
        raise HTTPException(404, "Page not found in this batch")

    i = ids.index(page_id)
    j = i - 1 if direction == "up" else i + 1
    if 0 <= j < len(ids):
        ids[i], ids[j] = ids[j], ids[i]
        db.reorder_pages(batch_id, ids)
    return JSONResponse({"order": ids})


@app.post("/batches/{batch_id}/translate")
def start_translate(batch_id: str):
    batch = db.get_batch(batch_id)
    if batch is None:
        raise HTTPException(404, "Batch not found")
    # One atomic UPDATE, not a separate read-then-write — two overlapping
    # requests for the same batch (a double-click, a retried request) can
    # otherwise both read "draft" before either has changed it, both pass,
    # and both enqueue a real run. See try_start_batch's docstring for what
    # that used to cause (the same batch processed twice, translations from
    # both runs rendered on top of each other).
    if not db.try_start_batch(batch_id):
        raise HTTPException(409, f"Batch is already {db.get_batch(batch_id)['status']}")
    enqueue_batch(batch_id)
    return RedirectResponse(f"/batches/{batch_id}", status_code=303)


@app.get("/batches/{batch_id}/status")
def batch_status(batch_id: str):
    batch = db.get_batch(batch_id)
    if batch is None:
        raise HTTPException(404, "Batch not found")
    return JSONResponse({"status": batch["status"], "error": batch["error"]})


# ---------------------------------------------------------------------------
# Image serving
# ---------------------------------------------------------------------------

@app.get("/pages/{page_id}/original")
def page_original(page_id: str):
    with db.get_conn() as conn:
        row = conn.execute("SELECT orig_path FROM pages WHERE id = ?", (page_id,)).fetchone()
    if row is None:
        raise HTTPException(404)
    return FileResponse(row["orig_path"])


@app.get("/pages/{page_id}/translated")
def page_translated(page_id: str):
    with db.get_conn() as conn:
        row = conn.execute("SELECT rendered_path FROM pages WHERE id = ?", (page_id,)).fetchone()
    if row is None or not row["rendered_path"]:
        raise HTTPException(404)
    return FileResponse(row["rendered_path"])


# ---------------------------------------------------------------------------
# Manual correction — fix a single block's translation, or its style
# (font size, position/size, colors), without re-running the whole
# (costly, non-deterministic) pipeline. See README's "Correcting a
# translation by hand".
# ---------------------------------------------------------------------------

def _rerender_page(page_id: str) -> None:
    page = db.get_page(page_id)
    if page is None:
        raise HTTPException(404, "Page not found")
    orig_image = Image.open(page["orig_path"])
    lines = db.list_lines(page_id)
    rendered = render_page(orig_image, lines)
    out_path = settings.outputs_dir / f"{page_id}.jpg"
    rendered.convert("RGB").save(out_path, format="JPEG", quality=92)
    db.set_page_rendered(page_id, str(out_path))


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return (int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16))


@app.post("/lines/{line_id}/edit")
async def edit_line(line_id: str, request: Request):
    """Body: {"translated_text": str, "fill_color": "#rrggbb"|null,
    "text_color": "#rrggbb"|null, "font_size": int|null, "manual_x"/
    "manual_y"/"manual_w"/"manual_h": int|null}. Only translated_text is
    required — the rest fall back to "no change"/"automatic" when omitted
    or null. Overwrites one block's translation/style and re-renders just
    its page — the fastest path from "this one line is wrong" to a fixed
    page, without waiting on (or paying for) a full re-translation of the
    batch.
    """
    line = db.get_line(line_id)
    if line is None:
        raise HTTPException(404, "Line not found")
    body = await request.json()
    translated_text = (body.get("translated_text") or "").strip()
    db.set_line_translation(line_id, translated_text, low_confidence=False)

    # Each of x/y/w/h is independently overridable — e.g. resizing without
    # repositioning is a normal thing to want, and previously required all
    # four to be set together or none of them took effect at all (a real
    # bug: editing only width/height, leaving x/y at their auto-filled
    # values, silently dropped the whole edit). See render.py's
    # layout_line for how a partial override now resolves against the
    # automatic layout for whichever fields aren't manually set.
    box_keys = ("manual_x", "manual_y", "manual_w", "manual_h")
    manual_box = tuple(
        int(body[k]) if body.get(k) not in (None, "") else None
        for k in box_keys
    )
    font_size = body.get("font_size")

    db.set_line_style(
        line_id,
        fill_color=_hex_to_rgb(body["fill_color"]) if body.get("fill_color") else None,
        text_color=_hex_to_rgb(body["text_color"]) if body.get("text_color") else None,
        manual_box=manual_box,
        manual_font_size=int(font_size) if font_size not in (None, "") else None,
    )
    _rerender_page(line["page_id"])
    return JSONResponse({"ok": True})


@app.get("/pages/{page_id}/debug")
def page_debug(page_id: str):
    """Per-page JSON dump of every detected line's real bbox/color/text (see
    worker.py's _write_debug_dump) — attach or paste this alongside a
    screenshot when a render looks wrong; it's what actually happened, not
    a guess from pixels."""
    with db.get_conn() as conn:
        row = conn.execute("SELECT id FROM pages WHERE id = ?", (page_id,)).fetchone()
    if row is None:
        raise HTTPException(404)
    debug_path = settings.outputs_dir / f"{page_id}_debug.json"
    if not debug_path.exists():
        raise HTTPException(404, "No debug info for this page yet — it's written when the batch finishes.")
    return FileResponse(debug_path, media_type="application/json")
