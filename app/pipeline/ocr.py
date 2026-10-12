"""
OCR, per detected text line — manga-ocr (kha-white/manga-ocr), a model
trained specifically on manga text rather than general-purpose
printed-document OCR.

Runs fully locally/offline (after its one-time model download from Hugging
Face on first use) and is CPU-friendly — no GPU required for a personal,
batch-at-a-time tool.

The `manga_ocr` import (and the torch/transformers stack it pulls in) is
deliberately lazy: importing this module doesn't require torch to be
installed, so the rest of the app works without it. It's only touched the
first time `run_ocr` actually executes.
"""
from __future__ import annotations

from PIL import Image, ImageOps

from .detection import DetectedLine

_mocr = None  # lazy singleton — loading the model is the expensive part


def _get_model():
    global _mocr
    if _mocr is None:
        from manga_ocr import MangaOcr  # local import: see module docstring
        _mocr = MangaOcr()
    return _mocr


def run_ocr(page_image: Image.Image, lines: list[DetectedLine]) -> list[str]:
    """Returns one OCR'd string per line, same order as `lines`."""
    model = _get_model()
    texts: list[str] = []
    for line in lines:
        x, y, w, h = line.bbox
        crop = page_image.crop((x, y, x + w, y + h)).convert("RGB")
        if line.text_color == (255, 255, 255):
            # manga-ocr is trained on normal (dark-text-on-light) polarity —
            # normalize inverted "whisper" lettering before handing it over.
            # (text_color, not fill_color, is the reliable signal here: fill
            # is now the actual sampled background color, not just black or
            # white — see detection.py's _estimate_local_fill.)
            crop = ImageOps.invert(crop)
        text = model(crop)
        texts.append(text.strip())
    return texts
