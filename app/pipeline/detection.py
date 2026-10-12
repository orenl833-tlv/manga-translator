"""
Text detection — the local pipeline stage that finds where the actual
lettering is, precisely, using a real trained text detector (CRAFT, via
EasyOCR — see `_get_detector`), then groups nearby detections into
bubble-level blocks (see `_merge_nearby_boxes`).

The underlying detector is genuinely the same category of technology
behind e.g. Google Translate's live camera-translate overlay — a
specialized text-detection model finding tight boxes around actual text
strokes — verified against how that works, not assumed. But a single
speech bubble routinely contains *several* of the detector's raw boxes:
vertical Japanese text is normally set as multiple narrow columns side by
side within one bubble, and the detector (correctly) finds each column as
its own box. Treating each of those narrow columns as an independent
render target — as Google's overlay does, and as an earlier version of
this file did — doesn't hold up here: a narrow column sized for one
character's width has nowhere near enough room for the wider horizontal
English replacement, and neighboring columns sit close enough together
that the tiny amount of breathing room between them isn't enough either.
The result was translated text overflowing its own column and visibly
running into the column next to it.

This is also, concretely, how every real working manga translator handles
this (confirmed by looking at several — zyddnys/manga-image-translator,
comic-translate, koharu — not assumed): they all detect/segment at the
*bubble* level and place one translation per bubble, not one per raw
detected line. `_merge_nearby_boxes` is the lightweight version of that
here: proximity-only geometry over boxes the detector already found (two
boxes get merged into one bounding rectangle if they sit close enough,
side by side or stacked, to plausibly be columns/lines of the same
bubble) — not a return to classifying a region as a "bubble" via
classical-CV contour heuristics, not shape/ellipse fitting, and not
inpainting. Those were real, working code at various points in this app's
history, and were also the actual source of nearly every rendering bug it
had; this only ever produces a plain rectangle, same as before.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

ROW_GAP_FRAC = 0.6  # y-gap (relative to box height) that starts a new reading-order "row"

Box = tuple[int, int, int, int]  # x, y, w, h


@dataclass
class DetectedLine:
    bbox: Box
    fill_color: tuple[int, int, int]
    text_color: tuple[int, int, int]


def _reading_order(items: list[tuple[Box, int]]) -> list[int]:
    """Right-to-left, top-to-bottom — the manga convention. Returns indices
    into `items` in reading order."""
    if not items:
        return []
    avg_h = sum(b[3] for b, _ in items) / len(items)
    by_y = sorted(items, key=lambda it: it[0][1] + it[0][3] / 2)

    rows: list[list[tuple[Box, int]]] = []
    for item in by_y:
        cy = item[0][1] + item[0][3] / 2
        if rows:
            last_row_cy = sum(b[1] + b[3] / 2 for b, _ in rows[-1]) / len(rows[-1])
            if abs(cy - last_row_cy) <= ROW_GAP_FRAC * avg_h:
                rows[-1].append(item)
                continue
        rows.append([item])

    ordered: list[int] = []
    for row in rows:
        row.sort(key=lambda it: it[0][0] + it[0][2] / 2, reverse=True)  # right to left
        ordered.extend(idx for _, idx in row)
    return ordered


# ---------------------------------------------------------------------------
# A trained neural text detector: find actual text strokes, anywhere on the
# page — the same job (and the same category of model) EAST/CRAFT-based
# systems do for this in general, per the patents describing how Google's
# live-camera translate overlay works.
#
# This runs CRAFT (Character Region Awareness for Text detection) via
# EasyOCR's `Reader(..., recognizer=False)` — detection only, none of
# EasyOCR's own text *recognition* is used (manga-ocr remains this app's
# OCR engine; see ocr.py). EasyOCR is the maintained path to CRAFT: the
# original standalone `craft-text-detector` PyPI package is unmaintained
# (last released 2022) and no longer loads on current torch/torchvision —
# confirmed by hand, not assumed.
# ---------------------------------------------------------------------------

_detector = None


def _get_detector():
    global _detector
    if _detector is None:
        import torch
        import easyocr
        # lang_list is required by the constructor but unused here — only
        # the detector (language-agnostic) is loaded, not any recognizer.
        _detector = easyocr.Reader(["en"], gpu=torch.cuda.is_available(),
                                    recognizer=False, verbose=False)
    return _detector


def _iou(a: Box, b: Box) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


_DUPLICATE_IOU_THRESHOLD = 0.35


def _drop_duplicate_boxes(boxes: list[Box]) -> list[Box]:
    """EasyOCR's `detect()` returns two separate lists — near-horizontal
    boxes and free-form (rotated/vertical) quads — and `_neural_text_boxes`
    combines both, since manga text can legitimately land in either. The
    two lists aren't always disjoint, though: the same physical text can
    get picked up in both (typically as very similar, heavily-overlapping
    boxes), which without this would become two separate "lines" — each
    independently OCR'd and independently translated, so each ends up
    reading as a different, equally-plausible phrasing of the same
    content, and both get drawn on top of each other. This is standard
    NMS-style dedup, not the bubble/shape logic this app deliberately
    avoids (see module docstring): among boxes that overlap heavily (IoU
    above `_DUPLICATE_IOU_THRESHOLD`), keep only the larger one. Two
    genuinely distinct lines sitting near each other — the normal case —
    share little to no area and are untouched by this.
    """
    boxes_by_area = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept: list[Box] = []
    for box in boxes_by_area:
        if all(_iou(box, k) < _DUPLICATE_IOU_THRESHOLD for k in kept):
            kept.append(box)
    return kept


def _neural_text_boxes(image_bgr: np.ndarray) -> list[Box]:
    """Returns axis-aligned boxes for every text region the detector finds
    — both the near-horizontal boxes it fits directly and the free-form
    (e.g. rotated/vertical) quads, reduced to their bounding rect, with
    near-duplicate boxes between the two collapsed down to one (see
    `_drop_duplicate_boxes`)."""
    reader = _get_detector()
    horizontal_lists, free_lists = reader.detect(image_bgr)
    boxes: list[Box] = []

    # A flat 4px cutoff is resolution-dependent — on a high-res scan, a
    # genuine 4x4 detection is essentially always noise (a stray edge in
    # line art, a screentone dot), not legible text. Scaling the floor to
    # the page's own size keeps it meaningful across scan resolutions.
    h_img, w_img = image_bgr.shape[:2]
    min_side = max(4, int(0.004 * min(h_img, w_img)))

    for x_min, x_max, y_min, y_max in horizontal_lists[0]:
        w, h = x_max - x_min, y_max - y_min
        if w >= min_side and h >= min_side:
            boxes.append((int(x_min), int(y_min), int(w), int(h)))

    for poly in free_lists[0]:
        pts = np.asarray(poly)
        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
        x2, y2 = pts[:, 0].max(), pts[:, 1].max()
        w, h = x2 - x1, y2 - y1
        if w >= min_side and h >= min_side:
            boxes.append((int(x1), int(y1), int(w), int(h)))

    return _drop_duplicate_boxes(boxes)


def _dominant_color(pixels: np.ndarray, bin_size: int = 16) -> tuple[int, int, int]:
    """The most common color among `pixels` (Nx3), not the median.

    A thin sampling ring immediately around a line of text can straddle a
    real edge — the text's own anti-aliased glyph border, or a bubble's
    drawn outline just past it — and an anti-aliased edge in a real
    (scanned/compressed) image isn't a clean two-color split, it's a short
    gradient of genuinely intermediate pixel values. When enough of the
    ring lands on that gradient, the *median* can land on one of those
    in-between values — a legitimate grey pixel that exists in the image,
    just not the color of either the text or its actual background. Colors
    are quantized into `bin_size`-wide buckets per channel first so minor
    compression noise doesn't fragment an otherwise-uniform background into
    many technically-distinct exact colors, each individually rare; the
    final value is the median of the pixels within the most common bucket
    (not the bucket's own quantized corner), which stays a real observed
    color rather than an average.
    """
    quantized = (pixels // bin_size) * bin_size
    values, counts = np.unique(quantized, axis=0, return_counts=True)
    dominant_bucket = values[np.argmax(counts)]
    in_bucket = pixels[np.all(quantized == dominant_bucket, axis=1)]
    return tuple(int(round(v)) for v in np.median(in_bucket, axis=0))


def _estimate_local_fill(image_bgr: np.ndarray, bbox: Box, pad_frac: float = 0.18) -> tuple[
    tuple[int, int, int], tuple[int, int, int]
]:
    """The actual color of the area immediately around a line of text,
    excluding the text itself — what the flat block render.py draws should
    match, not just "roughly light" or "roughly dark" reduced to pure
    black/white. There's no bubble contour to sample the fill from anymore
    (see module docstring), just whatever's directly behind this specific
    line, e.g. a tinted caption box or a grey panel background.

    `pad_frac` is deliberately modest (not "well past the text's own
    size"), so the sample stays close to what's actually immediately
    behind the text instead of reaching into a neighboring, differently
    colored area — and is capped in absolute pixels too, since a small
    bubble's own outline stroke (or the art just past it) can otherwise
    end up inside the sampled ring for a small text line even at a modest
    fraction of its own size. The sample is reduced with `_dominant_color`
    (the most common color, not the mean or the plain median — see that
    function's docstring for why the median specifically can occasionally
    land on a stray grey that doesn't match either the text or its real
    background). Text color itself stays a plain black/white choice (by
    the sampled fill's luminance) since that's purely about legibility,
    not matching the original page.
    """
    h, w = image_bgr.shape[:2]
    x, y, bw, bh = bbox
    pad_x = min(int(bw * pad_frac) + 4, 14)
    pad_y = min(int(bh * pad_frac) + 4, 14)
    rx1, ry1 = max(0, x - pad_x), max(0, y - pad_y)
    rx2, ry2 = min(w, x + bw + pad_x), min(h, y + bh + pad_y)

    region_mask = np.zeros((h, w), np.uint8)
    region_mask[ry1:ry2, rx1:rx2] = 255
    text_mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(text_mask, (x, y), (x + bw, y + bh), 255, thickness=cv2.FILLED)
    sample_mask = cv2.bitwise_and(region_mask, cv2.bitwise_not(text_mask))
    surrounding = image_bgr[sample_mask == 255]

    if surrounding.size:
        b, g, r = _dominant_color(surrounding.reshape(-1, 3))
    else:
        b, g, r = 255, 255, 255
    fill = (r, g, b)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    text_color = (0, 0, 0) if luminance >= 128 else (255, 255, 255)
    return fill, text_color


_MERGE_ALIGN_FRAC = 0.3   # how much of the shorter box's extent must overlap to count as "the same row/column band"
_MERGE_GAP_FRAC = 0.6     # max gap between two boxes, relative to a page's own typical box size, to still merge them
_MERGE_MAX_FRAC = 0.4     # a merged block can never exceed this fraction of the page's width/height


def _union_bbox(boxes: list[Box]) -> Box:
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[0] + b[2] for b in boxes)
    y2 = max(b[1] + b[3] for b in boxes)
    return (x1, y1, x2 - x1, y2 - y1)


def _should_merge(a: Box, b: Box, max_gap: float) -> bool:
    """Would `a` and `b` plausibly be two columns/lines of the same speech
    bubble or caption? Checked both ways — side by side (common for
    multi-column vertical Japanese text) and stacked (a multi-line
    horizontal caption) — since either can be true depending on the
    original text's orientation.

    "Side by side" means their vertical extents overlap substantially
    (they sit in the same horizontal band) and the horizontal gap between
    them is at most `max_gap`; "stacked" is the same check with the axes
    swapped. Two boxes that are already overlapping in the relevant axis
    produce a negative "gap", which trivially passes the threshold —
    correct, since overlapping boxes should merge regardless of how much
    they overlap by.

    `max_gap` is a fixed value the caller computes once from the *page's*
    typical box size — deliberately not recomputed here from `a`/`b`'s own
    size. An earlier version scaled the gap tolerance to the two boxes
    being compared, which is fine for two original raw detections but
    breaks down across repeated merge passes: once two boxes merge, the
    resulting union is bigger, which raised the tolerance for the *next*
    merge, which let it reach an even more distant box, compounding with
    each pass — a handful of genuinely separate bubbles down a column
    could cascade into one giant block spanning most of the page. A gap
    tolerance fixed up front doesn't compound.

    This is pure geometry over already-detected boxes, not a judgment
    about what a bubble looks like — the same reason `_drop_duplicate_boxes`
    isn't the shape/contour classification this app avoids elsewhere.
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    y_overlap = max(0, min(ay + ah, by + bh) - max(ay, by))
    x_gap = max(ax, bx) - min(ax + aw, bx + bw)
    side_by_side = y_overlap >= _MERGE_ALIGN_FRAC * min(ah, bh) and x_gap <= max_gap

    x_overlap = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    y_gap = max(ay, by) - min(ay + ah, by + bh)
    stacked = x_overlap >= _MERGE_ALIGN_FRAC * min(aw, bw) and y_gap <= max_gap

    return side_by_side or stacked


def _merge_nearby_boxes(boxes: list[Box], page_w: int, page_h: int) -> list[Box]:
    """Collapse groups of nearby boxes (see `_should_merge`) down to one
    bounding rectangle per group — repeated passes until nothing more
    merges, since merging two boxes can bring the resulting, larger
    rectangle close enough to a third to merge with that too. The page
    has at most a few dozen boxes, so the naive O(n^2)-per-pass approach
    is in no way a performance concern.

    The gap tolerance is a fixed value computed once from the *page's*
    typical box size (median width/height across all its raw detections),
    not from whatever's being compared on a given pass — see
    `_should_merge`'s docstring for why that matters. As a second,
    independent backstop, a merge is refused outright if the resulting
    box would exceed `_MERGE_MAX_FRAC` of the page's own width or height:
    a real speech bubble is never anywhere close to half a manga page, so
    a merge that would produce one is a sign something went wrong, not a
    real bubble.
    """
    if not boxes:
        return []
    sizes = sorted(dim for box in boxes for dim in (box[2], box[3]))
    typical = sizes[len(sizes) // 2]
    max_gap = typical * _MERGE_GAP_FRAC
    max_w, max_h = page_w * _MERGE_MAX_FRAC, page_h * _MERGE_MAX_FRAC

    groups = [[box] for box in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                box_i, box_j = _union_bbox(groups[i]), _union_bbox(groups[j])
                if not _should_merge(box_i, box_j, max_gap):
                    continue
                merged = _union_bbox([box_i, box_j])
                if merged[2] > max_w or merged[3] > max_h:
                    continue
                groups[i].extend(groups[j])
                del groups[j]
                changed = True
                break
            if changed:
                break
    return [_union_bbox(group) for group in groups]


def detect_lines(image_bgr: np.ndarray) -> list[DetectedLine]:
    """Every text block on the page, in manga reading order (right-to-left,
    top-to-bottom), each with its own precise box and estimated local
    fill/text color. "Block" here means one bubble/caption's worth of
    text — nearby raw detections (e.g. side-by-side columns of one
    bubble) are merged first (see `_merge_nearby_boxes`) — not one raw
    per-column/per-line detection; translate_shared.py is where a batch's
    worth of these blocks gets tied together for cross-block context
    (speaker identity, terminology), using the model's actual
    understanding of the page rather than a pixel-distance guess.
    """
    page_h, page_w = image_bgr.shape[:2]
    boxes = _merge_nearby_boxes(_neural_text_boxes(image_bgr), page_w, page_h)

    order = _reading_order([(box, i) for i, box in enumerate(boxes)])
    lines = []
    for i in order:
        bbox = boxes[i]
        fill_color, text_color = _estimate_local_fill(image_bgr, bbox)
        lines.append(DetectedLine(bbox=bbox, fill_color=fill_color, text_color=text_color))
    return lines
