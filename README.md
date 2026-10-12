


# Manga Translator

A self-hosted web app that translates whole batches of manga pages at once
and draws the translation back onto each page, in place of the original
lettering.

<!-- Replace with a GIF or a link to the demo video -->
> 🎬 **
https://github.com/user-attachments/assets/f14f4c3e-f020-4cba-8f0f-207f78c2a7fc
:** _add the 20-second demo video / GIF here_

## Why I built it

The best free option I found was Google Image Translate, and it falls short
for manga:

- **No batch mode.** You translate one page at a time.
- **Messy output.** Font sizes don't fit the bubbles and the text direction
  comes out wrong.
- **No consistency.** The same character's name can be translated
  differently from page to page.

This app fixes all three: you upload a chapter, every page is translated
in one pass with the whole batch as context, and you read the result in a
right-to-left manga reader.

## Features

- **Batch translation** with shared context across all pages, so speaker
  identity and character names stay consistent
- **Local computer vision:** a CRAFT text detector finds text regions, and
  nearby columns are merged into one block per speech bubble
- **Local OCR** with manga-ocr. Only the translation step calls an external API
- **Swappable LLM provider:** Anthropic (Claude) or OpenAI, chosen with
  a single env var
- **In-place rendering** that clips each translation to its own block, so
  text can't spill into neighboring bubbles
- **Manga reader** with real right-to-left navigation, cursor-anchored
  zoom, drag-to-pan and an original/translated toggle
- **Manual correction:** edit any line's text, font, colors or position
  and re-render that page instantly, with no extra API cost
- **Per-page debug view** that shows detection boxes, raw OCR and the
  translation, to tell detection, OCR and translation errors apart

## Tech stack

| Layer | Tools |
|---|---|
| Backend | Python, FastAPI, SQLite (job and page state), background worker |
| Computer vision | CRAFT text detection (via EasyOCR), manga-ocr, OpenCV, Pillow |
| LLM | Anthropic API / OpenAI API with structured output |
| Frontend | Jinja2 templates, vanilla JavaScript, CSS |
| Deployment | Docker (auto-uses a GPU if one is available, otherwise CPU) |

## How it works

```
upload pages → confirm reading order
  → detect text blocks per page (CRAFT, local)
  → OCR each block (manga-ocr, local)
  → translate ALL pages in one LLM request (shared context)
  → render each translation into its own block, clipped to its edges
  → read in the RTL reader, fix any line by hand
```

## Engineering highlights

Much of the work went into debugging, not adding features:

- **Overlapping text.** Japanese dialogue is set in narrow vertical columns,
  and each column was being translated and drawn separately, so the English
  text collided. I fixed it by merging columns into bubble-level blocks and
  rendering each block on its own clipped layer, which makes overflow
  impossible.
- **Silently untranslated pages.** Large batches hit the output-token
  limit, and only the first page or two got translated. The app now
  detects this and fails loudly instead.
- **An unmaintained dependency.** The standard CRAFT package no longer
  loads on current PyTorch. I switched to EasyOCR's maintained copy of the
  same detector.
- **Two redesigns.** I dropped an inpainting-based design and a
  badge-overlay design before landing on the current approach. It's
  modeled on how Google's camera translate overlay works.

The full write-up, including known limitations, is in
[docs/DESIGN.md](docs/DESIGN.md).

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # add ANTHROPIC_API_KEY (or set TRANSLATION_PROVIDER=openai + OPENAI_API_KEY)
uvicorn app.main:app --reload
```

Then open http://localhost:8000. The first run downloads the OCR and
detection models (~550 MB).

With Docker:

```bash
docker build -t manga-translator .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-... \
  -v manga-translator-data:/data manga-translator
```

## Note

A personal tool for manga you own. It doesn't publish or redistribute
anything, and it has no authentication, so keep it off the public
internet.

Built with [Claude Code](https://claude.com/claude-code).
