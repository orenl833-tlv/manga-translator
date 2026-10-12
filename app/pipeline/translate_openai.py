"""
Batch translation via the OpenAI Responses API — same job as
translate_anthropic.py (see that file for the design rationale), different
provider.

Uses `client.responses.create`, not the older Chat Completions endpoint —
verified against OpenAI's current API reference rather than assumed, since
the two endpoints use different shapes for both image content blocks and
function/tool calling and it's easy to mix them up by copying older
examples:
  - image content items are `{"type": "input_image", "image_url": <data
    URL as a plain string>, "detail": ...}` — not Chat Completions' nested
    `{"image_url": {"url": ...}}`.
  - text content items are `{"type": "input_text", "text": ...}`.
  - tool definitions are flat (`name`/`parameters` alongside `type`), not
    nested under a `"function"` key.
  - the system prompt goes in `input` as a `role: "developer"` message
    rather than the top-level `instructions` shortcut — equivalent, and
    keeps this file's shape consistent with translate_anthropic.py.
"""
from __future__ import annotations

import json

from openai import OpenAI

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

RECORD_TRANSLATIONS_TOOL = {
    "type": "function",
    "name": "record_translations",
    "description": "Record the translated text for every given text line.",
    "parameters": TRANSLATIONS_SCHEMA,
    "strict": True,
}


def _image_content(image) -> dict:
    return {
        "type": "input_image",
        "image_url": f"data:image/jpeg;base64,{encode_jpeg_b64(image)}",
        "detail": "auto",
    }


def translate_batch(pages: list[PageInput], target_lang: str) -> TranslationResult:
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set — see .env.example. The translation "
            "step is the one part of this pipeline that needs it."
        )

    client = OpenAI(api_key=settings.openai_api_key)

    prompt_pages, id_map = build_prompt_pages(pages)
    content: list[dict] = []
    for page in prompt_pages:
        content.append(_image_content(page.image))
        content.append({"type": "input_text", "text": page_text_block(page)})
    content.append({"type": "input_text", "text": checklist_text(prompt_pages)})

    response = client.responses.create(
        model=settings.openai_model,
        input=[
            {"role": "developer", "content": SYSTEM_PROMPT.format(target_lang=target_lang)},
            {"role": "user", "content": content},
        ],
        # A whole batch's worth of translations (every line, on every page)
        # comes back in one structured-output call — see the matching
        # comment in translate_anthropic.py for why too low a cap here
        # doesn't fail loudly, it just silently drops the later pages'
        # translations out of an otherwise "valid enough" truncated call.
        max_output_tokens=16000,
        tools=[RECORD_TRANSLATIONS_TOOL],
        tool_choice={"type": "function", "name": "record_translations"},
    )

    if response.status == "incomplete":
        reason = getattr(response.incomplete_details, "reason", None)
        raise RuntimeError(
            f"OpenAI's response was cut off before finishing (incomplete_details.reason={reason!r}) "
            "— likely too many pages/lines for one request. Try a smaller batch, or raise "
            "max_output_tokens in translate_openai.py."
        )

    call = next((item for item in response.output if item.type == "function_call"), None)
    if call is None:
        raise RuntimeError(f"OpenAI did not return a translation (status={response.status!r})")

    return parse_translations(json.loads(call.arguments), id_map)
