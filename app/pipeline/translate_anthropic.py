"""
Batch translation via the Claude API.

Design:
  - All pages in the batch are sent together, in ONE request, so the model
    can use the whole sequence as context (character identity, established
    terminology, who's speaking) rather than translating each line in
    isolation. This only holds up while a batch is small (a handful of
    pages) — see the README for how to scale past that.
  - Each page's *image* is sent alongside its OCR'd per-line text, not just
    the text. OCR can be wrong (especially for stylized lettering), and
    speaker attribution — critical for turning "Mayushi thinks..." into "I
    think..." — often depends on visual cues the text alone doesn't carry.
  - Line order within each page is already right-to-left/top-to-bottom
    (manga reading order) by the time it reaches here — see detection.py.
  - Output is forced through a strict tool schema rather than parsed out of
    free text, so there's no brittle "hope it's valid JSON" step.
"""
from __future__ import annotations

import anthropic

from ..config import settings
from .translate_shared import (
    SYSTEM_PROMPT,
    TRANSLATIONS_SCHEMA,
    PageInput,
    TranslationResult,
    build_prompt_pages,
    checklist_text,
    encode_jpeg_b64,
    page_text_block,
    parse_translations,
)

TRANSLATIONS_TOOL = {
    "name": "record_translations",
    "description": "Record the translated text for every given text line.",
    "input_schema": TRANSLATIONS_SCHEMA,
    "strict": True,
}


def _image_block(image) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": encode_jpeg_b64(image)}}


def translate_batch(pages: list[PageInput], target_lang: str) -> TranslationResult:
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set — see .env.example. The translation "
            "step is the one part of this pipeline that needs it."
        )

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    prompt_pages, id_map = build_prompt_pages(pages)
    content: list[dict] = []
    for page in prompt_pages:
        content.append(_image_block(page.image))
        content.append({"type": "text", "text": page_text_block(page)})
    content.append({"type": "text", "text": checklist_text(prompt_pages)})

    response = client.messages.create(
        model=settings.anthropic_model,
        # A whole batch's worth of translations (every line, on every page)
        # goes into one structured-output call, plus whatever reasoning
        # tokens `effort` spends getting there — both draw from this same
        # budget. Too low a cap doesn't fail loudly: the model fills in
        # translations page by page, so hitting the cap mid-response just
        # truncates the tail pages' entries out of an otherwise "valid
        # enough" partial tool call, silently translating only the first
        # page or two of a batch. 16000 comfortably covers several pages of
        # manga dialogue; see the stop_reason check below for what happens
        # if a genuinely large batch still exceeds it.
        max_tokens=16000,
        system=SYSTEM_PROMPT.format(target_lang=target_lang),
        output_config={"effort": settings.anthropic_effort},
        tools=[TRANSLATIONS_TOOL],
        tool_choice={"type": "tool", "name": "record_translations"},
        messages=[{"role": "user", "content": content}],
    )

    tool_use = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_use is None:
        raise RuntimeError(f"Claude did not return a translation (stop_reason={response.stop_reason!r})")
    if response.stop_reason == "max_tokens":
        # Whatever's in tool_use.input at this point is a partial response
        # cut off mid-generation — real for a large enough batch, and worth
        # failing loudly over: silently accepting it means some pages
        # (typically the later ones) come back with no translations at
        # all, with nothing in the UI to explain why. See README's
        # "Scaling past a few pages" for the actual fix (chunk the batch).
        raise RuntimeError(
            "Claude's response was cut off (stop_reason=max_tokens) before finishing every "
            "page in this batch — likely too many pages/lines for one request. Try a smaller "
            "batch, or raise max_tokens in translate_anthropic.py."
        )

    return parse_translations(tool_use.input, id_map)
