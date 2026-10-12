"""
Pieces of the translation step that don't depend on which LLM provider is
doing the actual translating — the data shapes, the prompt, and image
downsampling. See translate_anthropic.py / translate_openai.py for the
provider-specific request/response handling, and translate.py for how one
gets picked.

Detection (detection.py) finds precise per-bubble/caption text blocks with
a real trained detector, then groups nearby raw detections (e.g.
side-by-side columns of one bubble) into one block per bubble — see that
module's docstring for why grouping turned out to matter here. Each block
gets exactly one full, coherent translation; there's no splitting a
sentence across several ids the way an earlier version of this app (and
Google Translate's own live camera overlay, which translates each of its
detected blocks independently) had to. What's still worth doing beyond a
literal per-block translation is using the whole page/batch as *context*
— speaker identity, established terminology — even though each block's
own translation is self-contained.
"""
from __future__ import annotations

import base64
import io
import logging
import re
from dataclasses import dataclass, field

from PIL import Image

from ..config import settings

log = logging.getLogger("manga_translator.translate")

SYSTEM_PROMPT = """\
You are translating manga into {target_lang}.

You will be given a sequence of manga pages, in reading order, from one \
continuous scene/chapter. For each page you get the page image plus a list \
of text blocks already detected in manga reading order (right-to-left, \
top-to-bottom) with an id and the original-language OCR text for that \
block — each one is a full bubble or caption already (multi-column \
vertical text has already been joined into one block's OCR text), not a \
fragment of one.

For each block, give ONE complete, natural {target_lang} translation of \
what it's saying, the way a human translator would — not a literal \
word-for-word gloss.

Use ALL pages together as context, not just the block at hand:
- Track who is speaking. Japanese dialogue often refers to the speaker in \
third person or by name instead of "I" — once you can identify the \
speaker (from surrounding dialogue or the art: who's on screen, which \
character a bubble's tail points at), translate self-reference naturally \
into first person in {target_lang} instead of leaving it third-person, \
the way a human localizer would.
- Keep character names, terminology, and each character's voice/tone \
consistent across every page in this batch.
- OCR text can be wrong, especially for stylized lettering (sound effects, \
shouted text) — look at the image and use your judgement over the raw OCR \
when they conflict.
- For a sound effect/caption (not dialogue in a bubble), give a natural \
{target_lang} equivalent rather than a literal word-for-word gloss.
- translated_text must contain ONLY the rendered translation — nothing \
else. Never include brackets, asterisks, translator's notes, or \
uncertainty caveats like "[unclear]" or "*not relevant, ignore*" in \
translated_text: it gets drawn directly onto the page exactly as given, so \
any commentary there is drawn too. If a block's OCR text is garbled or \
you're unsure, give your best natural-sounding guess anyway (use the image \
for context) rather than describing the uncertainty in translated_text.
- Set low_confidence=true on an entry when you're genuinely guessing \
because the OCR text looks garbled/nonsensical and the image itself isn't \
legible enough there to confirm it — false otherwise. This is for a \
server log only, never shown on the page, so it costs nothing to flag \
honestly and it's the only place that uncertainty belongs; don't invent \
plausible-sounding specifics (names, numbers, dates) you can't actually \
read just to avoid flagging it.

Call record_translations with exactly one entry per block you were given \
(same page_id/line_id), never more, never fewer. Optionally include a \
short glossary of notable names/terms and how you translated them.
"""

# JSON Schema for the structured output both providers are forced to
# produce — Anthropic as a tool, OpenAI as a function — same shape either
# way so the rest of the app doesn't care which one ran.
#
# Strict-mode function/tool schemas (both providers) require every key
# listed under "properties" to also appear in "required" — there's no
# "optional key" the way plain JSON Schema allows; a field that's actually
# optional is instead typed nullable (`["string", "null"]`) and the model
# supplies null.
TRANSLATIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "page_id": {"type": "string"},
                    "line_id": {"type": "string"},
                    "translated_text": {"type": "string"},
                    "low_confidence": {"type": "boolean"},
                },
                "required": ["page_id", "line_id", "translated_text", "low_confidence"],
                "additionalProperties": False,
            },
        },
        "glossary": {
            "type": ["array", "null"],
            "items": {
                "type": "object",
                "properties": {
                    "term": {"type": "string"},
                    "translation": {"type": "string"},
                    "note": {"type": ["string", "null"]},
                },
                "required": ["term", "translation", "note"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["translations", "glossary"],
    "additionalProperties": False,
}

# Pinning the translations array to an exact expected length (minItems/
# maxItems) would be the obvious way to make an incomplete response
# impossible by construction — deliberately not done: both providers'
# strict/structured-output schemas reject minItems/maxItems on arrays
# beyond 0 or 1 (confirmed against current docs, not assumed), so adding
# either would 400 the request outright rather than help. Completeness is
# instead reinforced in the prompt (see `checklist_text`) and enforced
# after the fact (see `check_full_pages_present`).


@dataclass
class PageInput:
    page_id: str
    image: Image.Image
    # (line_id, source_text) — one per detected text line, in reading order
    lines: list[tuple[str, str]]


@dataclass
class TranslationResult:
    # (page_id, line_id) -> translated text
    translations: dict[tuple[str, str], str]
    glossary: list[dict[str, str]]
    low_confidence: list[tuple[str, str]] = field(default_factory=list)


def encode_jpeg_b64(image: Image.Image) -> str:
    """Downsample to the configured cap (image tokens are the dominant
    cost driver — see README) and return base64-encoded JPEG bytes."""
    img = image.convert("RGB")
    long_edge = max(img.size)
    if long_edge > settings.max_image_long_edge:
        scale = settings.max_image_long_edge / long_edge
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                          Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def page_text_block(page: PageInput) -> str:
    lines_desc = [f"Page {page.page_id} — text lines in reading order:"]
    if not page.lines:
        lines_desc.append("(no text detected on this page)")
    for line_id, source_text in page.lines:
        lines_desc.append(f"- line_id={line_id} ocr_text={source_text!r}")
    return "\n".join(lines_desc)


def checklist_text(prompt_pages: list[PageInput]) -> str:
    """A final, explicit list of every id the model is expected to cover —
    sent as the last message in the request, after every page's image/OCR
    text. The per-page instructions above already imply this, but spelling
    it out as a literal checklist (rather than something to infer from
    re-reading several pages of prior content) is a cheap, direct nudge
    against the model quietly wrapping up early on a demanding batch."""
    ids = [f"{page.page_id}/{line_id}" for page in prompt_pages for line_id, _ in page.lines]
    return (
        f"Checklist: call record_translations with EXACTLY one entry for each of these "
        f"{len(ids)} page_id/line_id pairs below, across all {len(prompt_pages)} pages — "
        "every one of them, no more, no fewer, even for a line you're unsure about (use "
        "low_confidence for that; never handle uncertainty by leaving an id out):\n"
        + ", ".join(ids)
    )


def build_prompt_pages(pages: list[PageInput]) -> tuple[list[PageInput], dict[tuple[str, str], tuple[str, str]]]:
    """Swap each page's real id (a 16-char UUID hex) and each line's real id
    (same) for short, sequential ones — p1, p2, ... / l1, l2, ... reset per
    page — before anything goes into a prompt.

    The model has to copy every id back verbatim for every single line in
    the batch — a real batch easily means dozens of long opaque hex
    strings in one call, which invites transcription slips. Short,
    sequential ids are far cheaper to carry correctly through a long
    response.

    Returns the id-swapped pages and a map back to the real (page_id,
    line_id) pairs for parse_translations to resolve the model's response
    against.
    """
    prompt_pages: list[PageInput] = []
    id_map: dict[tuple[str, str], tuple[str, str]] = {}
    for page_num, page in enumerate(pages, start=1):
        short_page_id = f"p{page_num}"
        short_lines: list[tuple[str, str]] = []
        for line_num, (line_id, text) in enumerate(page.lines, start=1):
            short_line_id = f"l{line_num}"
            id_map[(short_page_id, short_line_id)] = (page.page_id, line_id)
            short_lines.append((short_line_id, text))
        prompt_pages.append(PageInput(page_id=short_page_id, image=page.image, lines=short_lines))
    return prompt_pages, id_map


# Belt-and-suspenders for the "translated_text must contain only the
# translation" instruction above — instruction-following isn't guaranteed,
# and a model that leaks a bracketed aside gets it drawn onto the page
# verbatim.
_META_COMMENTARY_RE = re.compile(
    r"[\[(][^\[\]()]*\b(?:untranslat\w*|unclear|not relevant|ignore(?:d)? for translation|"
    r"translator'?s? note|no direct translation)\b[^\[\]()]*[\])]",
    re.IGNORECASE,
)


def _strip_meta_commentary(text: str) -> str:
    cleaned = _META_COMMENTARY_RE.sub("", text)
    cleaned = cleaned.replace("*", "").strip()
    if cleaned != text.strip():
        log.warning("stripped apparent meta-commentary out of a translated_text value: %r -> %r",
                    text, cleaned)
    return cleaned


def parse_translations(result: dict, id_map: dict[tuple[str, str], tuple[str, str]]) -> TranslationResult:
    translations: dict[tuple[str, str], str] = {}
    unresolved = 0
    low_confidence: list[tuple[str, str]] = []
    for item in result.get("translations", []):
        real = id_map.get((item["page_id"], item["line_id"]))
        if real is None:
            # The model returned an id pair that doesn't match anything we
            # gave it. Dropping it is deliberate: silently guessing which
            # real line it "must have meant" is how a translation ends up
            # rendered onto the wrong line — worse than leaving that line
            # untranslated and logged.
            unresolved += 1
            continue
        translations[real] = _strip_meta_commentary(item["translated_text"])
        if item.get("low_confidence"):
            low_confidence.append(real)
    if unresolved:
        log.warning("model returned %d translation(s) with an id we didn't recognize — "
                    "dropped rather than guessing which line they belong to", unresolved)
    if low_confidence:
        log.warning("model flagged %d translation(s) as low-confidence guesses "
                    "(likely garbled OCR source text): %s", len(low_confidence), low_confidence)
    glossary = result.get("glossary") or []
    return TranslationResult(translations=translations, glossary=glossary, low_confidence=low_confidence)


def check_full_pages_present(pages: list[PageInput], translations: dict[tuple[str, str], str]) -> None:
    """Raise if an entire page came back with zero translations despite
    having lines that needed one.

    A handful of individual lines missing here and there (garbled OCR the
    model couldn't make a guess for, an id it dropped) is a real, tolerated
    limitation — see the README. A whole page with nothing at all is a
    different kind of failure: it's what a response that got cut short (or
    that the model quietly wrapped up early on) partway through a batch
    looks like from the caller's side, and letting a batch "succeed" with
    entire pages silently untouched is worse than failing it outright with
    a clear reason.
    """
    translated_page_ids = {page_id for page_id, _ in translations}
    empty_pages = [page.page_id for page in pages if page.lines and page.page_id not in translated_page_ids]
    if empty_pages:
        raise RuntimeError(
            f"The model returned zero translations for {len(empty_pages)} of "
            f"{len(pages)} page(s) in this batch (page id(s): {', '.join(empty_pages)}), "
            "despite each having text lines to translate — this is what a response cut "
            "short partway through the batch looks like, not an isolated missed line. "
            "Try a smaller batch (see README's \"Scaling past a few pages\")."
        )
