"""
Render translations back onto the page.

Each translated line gets its own precise box from a real trained text
detector (detection.py) — not a bubble shape, not a merged multi-line
caption, just a plain rectangle tightly sized to that one line of original
lettering. Rendering fills that rectangle (grown a little past its exact
extent, see below) with a flat color sampled from what's immediately
around the line — see detection.py's `_estimate_local_fill` — and draws
the translation into it.

This is deliberately the same, simpler thing Google Translate's live
camera-translate overlay actually does — verified, not assumed: a flat
"color block" over the detected text, not real inpainting/reconstruction
(Google's own patents on this describe exactly that; more elaborate
generative reconstruction, e.g. Pixel's Magic Eraser, is a *separate*,
newer feature they've talked about layering on top of this same
detect-and-overlay approach, not what the base version has ever done).
There's no bubble shape to fit a translation into here either: a per-line
box is already just a rectangle.

One wrinkle a per-line box brings: it's sized to the *original* lettering,
and a narrow vertical column of Japanese text can be much narrower than
the horizontal English replacement needs (a single word can be wider than
the whole original column). Rather than force a line break in the middle
of a word to stay inside that width, `_layout_text` lets the drawn block
grow past the detected box — bounded by the page's own edges *and* by
every other detected line's own tight box, so growing to fit one line's
text can't run into a neighboring line/bubble (see `_x_growth_bounds`/
`_y_growth_bounds` in render_page) — before ever resorting to that.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger("manga_translator.render")

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
    # Last resort — a bitmap font with no size control. Bundle a .ttf under
    # app/static/fonts/ and point _FONT_CANDIDATES at it if this triggers.
    return ImageFont.load_default()


def _break_long_word(draw: ImageDraw.ImageDraw, word: str, font: ImageFont.FreeTypeFont,
                      max_width: int) -> list[str]:
    """Split a single token character-by-character so it still respects
    max_width — word-wrapping alone can't help a token with no spaces to
    break on (a long name, "...", a run-on with no punctuation)."""
    if len(word) <= 1 or draw.textlength(word, font=font) <= max_width:
        return [word]
    chunks: list[str] = []
    current = ""
    for ch in word:
        trial = current + ch
        if not current or draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            chunks.append(current)
            current = ch
    if current:
        chunks.append(current)
    return chunks


def _wrap_to_fit(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont,
                  max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = ""
    for word in words:
        if draw.textlength(word, font=font) > max_width:
            if current:
                lines.append(current)
                current = ""
            lines.extend(_break_long_word(draw, word, font, max_width))
            continue
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _wrap_whole_words(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont,
                       max_width: int) -> list[str] | None:
    """Like `_wrap_to_fit`, but returns None instead of ever splitting a
    word's characters apart — the caller widens `max_width` and retries
    when this fails, rather than accepting a mid-word break."""
    words = text.split()
    if not words:
        return [""]
    if any(draw.textlength(word, font=font) > max_width for word in words):
        return None
    lines: list[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


# A detected line's own box is sized to the *original* lettering — for
# vertical Japanese text that's a narrow, tall column, which occasionally
# has too little width for even the smallest readable English replacement
# (a long word can be wider than the entire original column at any font
# size). Growing the block past the original box is only a last resort for
# that case, not the normal path: growing it routinely is what used to
# blow text out past a round bubble's edge and into neighboring lines, so
# the caps here are deliberately modest.
_MAX_BLOCK_WIDTH_MULT = 2.2
_MAX_BLOCK_HEIGHT_MULT = 1.5


def _layout_text(draw: ImageDraw.ImageDraw, text: str, base_w: float, base_h: float,
                  max_w: float, max_h: float, max_size: int = 26, min_size: int = 9,
                  ) -> tuple[list[str], ImageFont.FreeTypeFont, float, float]:
    """Pick a font size and line-wrapping for `text` that fits inside the
    original `base_w`/`base_h` box — a real speech bubble is sized with
    margin around its text, so the detected line's own (padded) box is
    normally enough room, and shrinking the font to fit it is what keeps
    the result inside the bubble instead of bursting past its edge.

    Only once no font size down to `min_size` fits the original box without
    splitting a word does this grow the block at all, and then only up to
    `max_w`/`max_h` (bounded by the page's edges — see render_page); a
    character-level split (`_wrap_to_fit`) is the very last resort, for a
    single token too long to fit even at the smallest font and widest
    allowed block.
    """
    for size in range(max_size, min_size - 1, -2):
        font = _load_font(size)
        line_h = font.getbbox("Ag")[3] + 4
        lines = _wrap_whole_words(draw, text, font, base_w)
        if lines is not None and line_h * len(lines) <= base_h:
            return lines, font, base_w, base_h

    font = _load_font(min_size)
    line_h = font.getbbox("Ag")[3] + 4
    width = base_w
    while True:
        lines = _wrap_whole_words(draw, text, font, width)
        if lines is not None:
            total_h = line_h * len(lines)
            if total_h <= max_h:
                return lines, font, max(width, base_w), max(total_h, base_h)
        if width >= max_w:
            break
        width = min(width * 1.3, max_w)

    lines = _wrap_to_fit(draw, text, font, max_w)
    # A genuinely pathological case (a long phrase with no good break points
    # into a tiny original line) can still need more height than max_h
    # allows — clamp the block itself to what the page actually has room
    # for rather than requesting more; the text may then run past the
    # block's own edge in that rare case, which is a smaller visual defect
    # than asking render_page to place a block bigger than the page.
    return lines, font, max_w, max(min(line_h * len(lines), max_h), base_h)


def _draw_lines(draw: ImageDraw.ImageDraw, block: tuple[float, float, float, float],
                 lines: list[str], font: ImageFont.FreeTypeFont, color: tuple[int, int, int]) -> None:
    x, y, w, h = block
    line_h = font.getbbox("Ag")[3] + 4
    total_h = line_h * len(lines)
    ty = y + max(0, (h - total_h) / 2)

    # An outline in the opposite color is cheap insurance against a wrong
    # (or borderline) fill/text polarity call — the flat block is filled
    # with an estimate of the surrounding color, not a certainty.
    font_size = getattr(font, "size", 16)
    stroke_width = max(1, round(font_size * 0.07))
    luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
    stroke_color = (0, 0, 0) if luminance > 127 else (255, 255, 255)

    for line in lines:
        line_w = draw.textlength(line, font=font)
        tx = x + max(0, (w - line_w) / 2)
        draw.text((tx, ty), line, font=font, fill=color,
                  stroke_width=stroke_width, stroke_fill=stroke_color)
        ty += line_h


def _clamp_origin(center: float, size: float, lo: float, hi: float) -> float:
    hi = max(lo, hi - size)
    return min(max(center - size / 2, lo), hi)


Box = tuple[float, float, float, float]  # x, y, w, h


def _x_growth_bounds(this_box: Box, other_boxes: list[Box], lo: float, hi: float) -> tuple[float, float]:
    """How far a block centered on `this_box` can grow horizontally without
    crossing into another *detected line's own tight box* — not that
    line's own (possibly also grown) rendered block, since two lines
    growing toward each other from opposite sides would otherwise each be
    "avoiding" a moving target and could still collide in the middle.
    Instead, each stops at the midpoint between its own tight edge and the
    neighbor's tight edge, so two lines that both grow toward one another
    meet at that midpoint rather than passing through it.

    Only a neighbor whose own vertical extent overlaps this line's counts
    — one sitting in a different row/panel entirely doesn't constrain
    horizontal growth here.
    """
    x, y, w, h = this_box
    left, right = lo, hi
    for ox, oy, ow, oh in other_boxes:
        if oy + oh <= y or oy >= y + h:
            continue
        if ox >= x + w:
            right = min(right, ((x + w) + ox) / 2)
        elif ox + ow <= x:
            left = max(left, (x + (ox + ow)) / 2)
    return left, right


def _y_growth_bounds(this_box: Box, other_boxes: list[Box], lo: float, hi: float) -> tuple[float, float]:
    """Vertical counterpart to `_x_growth_bounds` — bounds growth against
    neighbors whose horizontal extent overlaps this line's own."""
    x, y, w, h = this_box
    top, bottom = lo, hi
    for ox, oy, ow, oh in other_boxes:
        if ox + ow <= x or ox >= x + w:
            continue
        if oy >= y + h:
            bottom = min(bottom, ((y + h) + oy) / 2)
        elif oy + oh <= y:
            top = max(top, (y + (oy + oh)) / 2)
    return top, bottom


def layout_line(draw: ImageDraw.ImageDraw, r: sqlite3.Row, other_boxes: list[Box],
                 img_w: int, img_h: int, margin: int = 4,
                 ) -> tuple[Box, ImageFont.FreeTypeFont, list[str]]:
    """The block/font/wrapped-lines a line's translation renders with right
    now — the automatic layout, with any manual override applied on top.
    Factored out of render_page so other callers (the correction UI's
    "what's the current width/position" display — see app/main.py) can ask
    the same question `render_page` answers when it actually draws,
    without a second, drifting copy of this logic.

    x/y/w/h/font_size are each independently overridable — resizing
    without repositioning (or vice versa) is a normal thing to want. An
    earlier version treated the box as one all-or-nothing unit (manual_x
    set meant "use manual_w/manual_h too, no matter what"), which silently
    ignored a lone width/height override whenever x/y hadn't also been
    set manually.
    """
    text = (r["translated_text"] or "").strip()
    manual_font = r["manual_font_size"]
    manual_x, manual_y = r["manual_x"], r["manual_y"]
    manual_w, manual_h = r["manual_w"], r["manual_h"]

    x, y, w, h = r["x"], r["y"], r["w"], r["h"]
    cx, cy = x + w / 2, y + h / 2
    # Pad a little past the line's exact detected extent so the flat
    # block fully covers the original glyphs, including any
    # anti-aliased edge pixels just outside the tight box.
    pad_x, pad_y = int(w * 0.12) + 2, int(h * 0.12) + 2
    base_w, base_h = w + 2 * pad_x, h + 2 * pad_y

    left, right = _x_growth_bounds((x, y, w, h), other_boxes, margin, img_w - margin)
    top, bottom = _y_growth_bounds((x, y, w, h), other_boxes, margin, img_h - margin)
    # Never shrink below the line's own (padded) box just because a
    # neighbor sits close — that would make an already-tight box
    # impossibly small. In that rare case the block may end up
    # slightly touching a neighbor's box; still far better than
    # ballooning through it uncapped.
    left, right = min(left, cx - base_w / 2), max(right, cx + base_w / 2)
    top, bottom = min(top, cy - base_h / 2), max(bottom, cy + base_h / 2)

    if manual_x is None and manual_y is None and manual_w is None and manual_h is None and manual_font is None:
        # Fully automatic: fit the font, allow the block to grow past the
        # detected extent for a word that needs more room (see
        # _layout_text), and place it within the neighbor-aware bounds.
        max_w = min(base_w * _MAX_BLOCK_WIDTH_MULT, right - left)
        max_h = min(base_h * _MAX_BLOCK_HEIGHT_MULT, bottom - top)
        text_lines, font, block_w, block_h = _layout_text(draw, text, base_w, base_h, max_w, max_h)
        block_x = _clamp_origin(cx, block_w, left, right)
        block_y = _clamp_origin(cy, block_h, top, bottom)
        return (block_x, block_y, block_w, block_h), font, text_lines

    # Something's manual — resolve size first (falling back to the
    # automatic padded size for whichever of w/h isn't overridden), then
    # position (falling back to centering that size within the same
    # neighbor-aware bounds the automatic layout itself respects).
    block_w = manual_w if manual_w is not None else base_w
    block_h = manual_h if manual_h is not None else base_h
    block_x = manual_x if manual_x is not None else _clamp_origin(cx, block_w, left, right)
    block_y = manual_y if manual_y is not None else _clamp_origin(cy, block_h, top, bottom)

    if manual_font is not None:
        font = _load_font(manual_font)
        text_lines = _wrap_to_fit(draw, text, font, block_w)
    else:
        # Fit the font to whatever box is actually in effect now — a
        # manually-resized box gets its own font auto-fit, not whatever
        # size the automatic box (which may be a different shape
        # entirely) happened to pick — capped at exactly this box's own
        # size, no further growth, since a manual box means the user has
        # already taken over sizing. Text that still doesn't fit is
        # clipped (see _paint_block) rather than overlapping a neighbor.
        text_lines, font, block_w, block_h = _layout_text(
            draw, text, block_w, block_h, block_w, block_h)

    return (block_x, block_y, block_w, block_h), font, text_lines


def render_page(orig_image: Image.Image, lines: list[sqlite3.Row]) -> Image.Image:
    img = orig_image.convert("RGB")

    translated = [r for r in lines if (r["translated_text"] or "").strip()]
    untranslated = [r for r in lines if not (r["translated_text"] or "").strip()]
    if untranslated:
        # The model was asked for exactly one entry per given line — if some
        # came back empty, that's worth knowing about rather than silently
        # leaving the original text on the page. Not retried automatically
        # (see README's known limitations); at least it's visible in logs
        # and a page's debug info.
        log.warning("page has %d line(s) with no translation: %s", len(untranslated),
                    [r["id"] for r in untranslated])
    if not translated:
        return img

    draw = ImageDraw.Draw(img)
    img_w, img_h = img.size
    margin = 4
    # Every OTHER detected line's own tight box (translated or not — an
    # untranslated line still has its original, untouched lettering sitting
    # there) is a hard exclusion zone: a block is allowed to grow past its
    # own detected extent (see layout_line), but never into a neighboring
    # line's territory. This is the difference between the earlier version
    # of this fix (a flat growth cap, page-edges-only) and this one: two
    # lines that are simply close together — the common case for stacked
    # manga bubbles — no longer grow into each other.
    all_boxes: list[tuple[str, Box]] = [(r["id"], (r["x"], r["y"], r["w"], r["h"])) for r in lines]

    for r in translated:
        fill = tuple(json.loads(r["fill_color"]))
        text_color = tuple(json.loads(r["text_color"]))
        others = [box for line_id, box in all_boxes if line_id != r["id"]]
        block, font, text_lines = layout_line(draw, r, others, img_w, img_h, margin)
        _paint_block(img, block, fill, text_lines, font, text_color)

    return img


def _paint_block(img: Image.Image, block: tuple[float, float, float, float],
                  fill: tuple[int, int, int], lines: list[str], font: ImageFont.FreeTypeFont,
                  text_color: tuple[int, int, int]) -> None:
    """Paint one translated block: a rounded-corner fill with its text
    drawn into it, composited onto `img` so that neither can ever land
    outside the block's own rectangle.

    Earlier this drew directly onto the page — `_layout_text`/the
    neighbor-growth bounds size a block to fit its text, but that's a
    computed estimate, not a guarantee, and Pillow doesn't clip text to
    any boundary on its own. Wherever that estimate was off (a font/line-
    height rounding difference, a genuinely pathological too-long phrase),
    the actual glyphs could run past the block's edge and into whatever
    was there — usually a neighboring block, reported as text overlapping
    text. Painting onto a separate layer exactly the block's own size and
    compositing just that layer makes overflow physically impossible
    instead of merely unlikely: Pillow silently discards anything drawn
    past a fixed-size image's own edges.
    """
    x, y, w, h = block
    ix, iy = int(round(x)), int(round(y))
    iw, ih = max(1, int(round(w))), max(1, int(round(h)))

    layer = Image.new("RGBA", (iw, ih), (0, 0, 0, 0))
    layer_draw = ImageDraw.Draw(layer)
    # Rounded rather than square corners: a real speech bubble is
    # round/oval, and a sharp rectangle corner is the part of the block
    # most likely to visibly poke past its curve. This doesn't fit the
    # bubble's actual outline (see detection.py's module docstring on why
    # that isn't attempted), just softens the one part of a plain
    # rectangle most likely to stick out of it.
    radius = min(iw, ih) * 0.25
    layer_draw.rounded_rectangle([0, 0, iw, ih], radius=radius, fill=(*fill, 255))
    _draw_lines(layer_draw, (0, 0, iw, ih), lines, font, text_color)

    img.paste(layer, (ix, iy), mask=layer)
