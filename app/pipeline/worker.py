"""
Ties the pipeline stages together for one batch, and runs batches in the
background so the upload/translate HTTP request returns immediately.

This is a single-user tool, so the job runner is deliberately minimal: one
background worker thread, one batch at a time. No Redis, no Celery — that's
infra this app doesn't need at this scale. Progress is still visible across
page reloads because every stage writes its result to SQLite as it goes.
"""
from __future__ import annotations

import json
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image

from .. import db
from ..config import settings
from .detection import detect_lines
from .ocr import run_ocr
from .render import render_page
from .translate import PageInput, translate_batch

log = logging.getLogger("manga_translator.worker")

# Single worker: batches are processed one at a time. Fine for a personal
# tool; raise max_workers if you want concurrent batches on beefier hardware.
_executor = ThreadPoolExecutor(max_workers=1)


def enqueue_batch(batch_id: str) -> None:
    # Callers are expected to have already flipped the batch to "queued"
    # via db.try_start_batch's atomic check-and-set (see main.py) — that's
    # what actually prevents the same batch being submitted twice, not
    # anything here.
    _executor.submit(_run_batch, batch_id)


def _run_batch(batch_id: str) -> None:
    try:
        db.set_batch_status(batch_id, "processing")
        _process(batch_id)
        db.set_batch_status(batch_id, "done")
    except Exception as exc:  # noqa: BLE001 — surface any failure to the UI
        log.exception("batch %s failed", batch_id)
        db.set_batch_status(batch_id, "error", error=f"{exc}\n{traceback.format_exc()}")


def _process(batch_id: str) -> None:
    batch = db.get_batch(batch_id)
    pages = db.list_pages(batch_id)

    page_inputs: list[PageInput] = []
    page_images: dict[str, Image.Image] = {}

    for page in pages:
        image = Image.open(page["orig_path"])
        page_images[page["id"]] = image

        # A (re)processing pass is a redo, not an append — see
        # clear_lines's docstring for why this matters even with
        # try_start_batch's concurrency guard in place.
        db.clear_lines(page["id"])

        image_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        detected = detect_lines(image_bgr)
        source_texts = run_ocr(image, detected) if detected else []

        line_tuples: list[tuple[str, str]] = []
        for order_index, (line, text) in enumerate(zip(detected, source_texts)):
            line_id = db.add_line(
                page_id=page["id"], order_index=order_index, bbox=line.bbox,
                fill_color=line.fill_color, text_color=line.text_color, source_text=text,
            )
            line_tuples.append((line_id, text))

        page_inputs.append(PageInput(page_id=page["id"], image=image, lines=line_tuples))

    # One request for the whole batch — see translate.py for why.
    result = translate_batch(page_inputs, target_lang=batch["target_lang"])
    db.set_batch_glossary(batch_id, result.glossary)

    low_confidence = set(result.low_confidence)
    for (page_id, line_id), text in result.translations.items():
        db.set_line_translation(line_id, text, low_confidence=(page_id, line_id) in low_confidence)

    for page in pages:
        lines = db.list_lines(page["id"])
        _write_debug_dump(page, lines)
        rendered = render_page(page_images[page["id"]], lines)
        out_path = settings.outputs_dir / f"{page['id']}.jpg"
        rendered.convert("RGB").save(out_path, format="JPEG", quality=92)
        db.set_page_rendered(page["id"], str(out_path))


def _write_debug_dump(page, lines) -> None:
    """One JSON file per rendered page, next to it in outputs_dir, listing
    every detected line's real geometry/color/text — not just what a
    screenshot of the final render shows. Attach or paste this alongside a
    screenshot when something looks wrong; the real cause (a detection
    miss, bad OCR, or a translation mistake) is usually obvious from it
    instead of guessed at.
    """
    dump = {
        "page_id": page["id"],
        "filename": page["filename"],
        "width": page["width"],
        "height": page["height"],
        "lines": [
            {
                "id": r["id"],
                "order_index": r["order_index"],
                "bbox_xywh": [r["x"], r["y"], r["w"], r["h"]],
                "source_text": r["source_text"],
                "translated_text": r["translated_text"],
                "low_confidence": bool(r["low_confidence"]),
            }
            for r in lines
        ],
    }
    out_path = settings.outputs_dir / f"{page['id']}_debug.json"
    try:
        out_path.write_text(json.dumps(dump, indent=2, ensure_ascii=False))
    except OSError:
        log.exception("couldn't write debug dump for page %s", page["id"])
