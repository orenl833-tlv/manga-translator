# Manga Translator — design notes

The full technical write-up: how the pipeline works, why it ended up
this way, and every known limitation. For a quick overview, see the
[main README](../README.md).

Upload a batch of manga page images, confirm the reading order, pick a
target language, and get pages back with translated dialogue drawn
directly in place over the original lettering. Self-hosted: runs on your
own machine, GPU or not (auto-uses one if present, otherwise CPU).

**This is a personal exercise tool, not a distribution tool.** It renders
translated pages back to you in your own browser session; it doesn't
publish or share anything. Use it on manga you own, for yourself — see the
legal note at the bottom.

## How it works

```
upload batch → confirm page order → for each page:
    find every text block on the page (a trained neural detector — CRAFT,
    with nearby raw detections — e.g. side-by-side text columns of one
    bubble — merged into one block per bubble/caption)
    OCR each block (local, manga-ocr)
  → send ALL pages (images + every page's block OCR text) to the LLM in
    ONE request → one complete, natural translation per block, with the
    whole batch as shared context (speaker identity, terminology)
  → fill each block's own box with a flat color and draw its translation
    into it, in place — composited from a separate layer sized exactly to
    that block, so text can't spill past its own edges into a neighbor
```

Only the translation step calls out over the network (to whichever
provider is configured — see Configuration). Detection and OCR run
entirely on your machine.

### Why it's built this way

This went through two earlier designs before landing here, and knowing
why is useful if you're going to touch this code:

**Attempt 1: detect exact bubble shapes (classical computer vision),
OCR each one, and reconstruct the bubble's interior with an inpainting
model before drawing the translation into it.** This is the most
ambitious, most "properly done" version, and for real speech bubbles —
which have an actual drawn outline to erase into and refit against — it
mostly worked. For everything else (captions, titles, sound effects: text
with no bubble around it), it didn't: there's no real contour to fit into,
no reliably-known local fill color, and no way to tell "safely empty
space" apart from a piece of art or another region sitting right next to
it. Across many rounds of fixing real, reported bugs in that area, each
fix uncovered a new failure mode in the same place — a sign the foundation
itself (guessing precise geometry from pixels, with no way to verify the
guess) didn't support the reliability needed, however much it got tuned.

**Attempt 2: give up on in-panel placement entirely.** Have the
translation model find and read text itself (no local detection/OCR at
all) and give only a rough anchor point per region; render a small
numbered badge at each point and list every translation in a strip below
the page. This was completely reliable — nothing can overflow a box or
collide with a neighbor if nothing is drawn into the art — but it doesn't
actually look like a translated manga page anymore, which turned out to
defeat the point.

**What's here now is a third design, built by researching how a working
system actually solves this — Google Translate's own live camera
translate overlay — rather than re-guessing:** it's powered by a
specialized, purpose-trained text-detection model (the CRAFT/EAST family —
confirmed from Google's own patents on this feature, not assumed), not a
general vision-language model reasoning about coordinates, and it doesn't
attempt real inpainting/reconstruction — it draws a flat color block over
the detected text and puts the translation on top. Both of those turned
out to be exactly the right lessons:

- **Detection stays local, using a real trained detector (CRAFT), and it
  finds plain rectangular boxes — never a "bubble" shape via contour
  fitting.** This was actually the one part of this app's original
  pipeline that stayed reliable across every rendering rework in its
  history; what caused the bugs was everything that used to be built *on
  top of* that detector (classifying a region as a bubble via classical-CV
  contour heuristics, fitting text into a bubble's round/oval shape). None
  of that exists here.
- **Rendering fills a block's box with a flat, locally-sampled color and
  draws the translation directly into it — not real reconstruction.** This
  is simpler than the inpainting this app used to do, and it's genuinely
  what the real system it's modeled on does too, not a corner cut.
- **Nearby raw detections are merged into one block per bubble/caption
  before translation, not kept as independent per-column/per-line
  boxes.** This was added after the design above still produced
  overlapping, unreadable text in testing, once traced to its real cause:
  Japanese dialogue is normally set as several narrow vertical columns
  side by side within *one* bubble, the detector (correctly) finds each
  column as its own box, and a narrow column has nowhere near enough room
  for its horizontal English replacement — so translated text from
  adjacent columns ran into each other. Every real, working manga
  translator checked while diagnosing this (zyddnys/manga-image-translator,
  comic-translate, koharu) detects and places text at the *bubble* level
  for exactly this reason, not per raw line — this app's version of that
  is deliberately lightweight (proximity-only geometry merging boxes the
  detector already found into one bounding rectangle), not the contour
  classification or shape-fitting ruled out above. See
  `_merge_nearby_boxes` in `app/pipeline/detection.py`.
- **A translated block's fill and text are composited from a separate,
  exactly-sized layer, not drawn straight onto the page** — so a block's
  own rectangle is a hard boundary text can't cross, not just an estimate
  a layout calculation tries to respect. Whatever produced the overlap
  above, sizing every block generously enough to always be certain in
  advance isn't reliable; making overflow physically impossible is. See
  `_paint_block` in `app/pipeline/render.py`.
- **Translation still uses the whole page (and batch) as context** for
  things a literal per-block translation can't get right alone — speaker
  identity, consistent terminology — even though each block now gets one
  self-contained translation rather than a fragment split across several
  ids (an earlier version of this app tried the latter, matching a live
  camera overlay's per-block-independent translation but keeping
  cross-line context; merging fixed the actual positioning problem more
  directly, so the id-splitting complexity was no longer needed either).

Neither earlier attempt was abandoned lightly; each went through many
rounds of real bug fixes before the design changed.

**All pages in a batch are still translated in one request**, so the
model can use the whole sequence as context (character identity,
established terminology, who's speaking) rather than reading each page
blind. This is what resolves things like "Mayushi thinks Mayushi
should..." into a natural first-person line. This only holds up for
batches that fit comfortably in one request — see **Scaling past a few
pages** below.

**Text blocks are referred to by short ids (`p1`/`l2`, not a raw database
id) in the translation request/response**, reset per page. The model has
to echo an id back for every single block in the batch — dozens of them,
across every page, in one structured-output call — and a long opaque id is
real risk there: a single mis-copied character is indistinguishable from a
translation landing on the wrong block once it's rendered. Short, small,
sequential ids are far cheaper for a model to carry correctly through a
long response; anything it returns with an id that doesn't resolve is
dropped (and logged) rather than guessed at.

### A real dependency snag, and how it's handled

The obvious pip package for CRAFT is `craft-text-detector` — it's what
this app used first. It turned out to be unmaintained (last released
2022) and no longer loads on current torch/torchvision at all (confirmed
by hand: it imports a torchvision internal — `model_urls` — that no longer
exists, and patching past that hits a second, deeper incompatibility in
how torchvision builds its VGG backbone now). Rather than ship something
that doesn't actually run, detection.py uses **EasyOCR's detector**
instead (`easyocr.Reader(..., recognizer=False)`) — it's the same CRAFT
architecture, just under active maintenance, and it's confirmed working
end-to-end on the current dependency stack. EasyOCR's own text
*recognition* is never used here — manga-ocr remains the OCR engine.

One consequence: `pip install -r requirements.txt` will print dependency
resolver warnings (opencv-python vs. opencv-python-headless, a numpy
version pinned back by manga-ocr's own dependencies). These look alarming
but are non-fatal — every piece of this pipeline has been run end-to-end
against the exact combination in `requirements.txt` and works correctly
despite them.

## Running it

### Locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
uvicorn app.main:app --reload
```

Then open http://localhost:8000. `pip install torch` (a `requirements.txt`
dependency, via manga-ocr) picks up a GPU build automatically if CUDA is
available on your machine, and falls back to CPU otherwise — nothing to
configure either way.

The first run downloads two models (manga-ocr and the CRAFT text
detector — roughly 450MB + 100MB); after that it's fully offline except
for the translation API call. There's no third, inpainting model anymore
— rendering doesn't need one (see "Why it's built this way").

### Docker

```bash
docker build -t manga-translator .
docker run -p 8000:8000 \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -v manga-translator-data:/data \
  manga-translator
```

For GPU passthrough under Docker, add `--gpus all` to `docker run` (needs
the NVIDIA Container Toolkit on the host) — the app picks it up
automatically, same as running locally.

## Configuration

See `.env.example` for all options. Worth knowing about:

- `TRANSLATION_PROVIDER` (default `anthropic`) — set to `openai` to use
  OpenAI's API instead (needs `OPENAI_API_KEY`/`OPENAI_MODEL` set instead
  of the Anthropic ones). Only that one file's worth of code differs
  between providers — see `app/pipeline/translate_anthropic.py` /
  `translate_openai.py` — everything else in the pipeline is unaffected by
  this choice.
- `ANTHROPIC_MODEL` (default `claude-sonnet-5`) — cheaper than Opus-tier
  and plenty capable for this; bump it if translation quality matters more
  to you than cost.
- `MAX_IMAGE_LONG_EDGE` (default `1600`) — images are downsampled to this
  before being sent to the translation model. Image tokens are the
  dominant cost driver; this is the main lever if you want to spend less
  per page. OCR itself runs locally at full resolution and is unaffected
  by this setting.

## Scaling past a few pages

The one-request-per-batch design is intentional for small batches (a
handful of pages) — it's simpler and gives strictly better cross-page
coherence than any incremental-context scheme, at negligible cost. It
doesn't scale indefinitely: translating an entire volume in one request
would eventually hit context/cost/latency limits. If you want to translate
much larger batches, the natural extension is to chunk into groups of
~10-15 pages and carry forward a compact glossary (the `glossary` a batch
already returns is exactly this) into the next chunk's system prompt,
instead of one unbounded request. Not implemented here — this app is sized
for "a chapter at a time."

The one request is also capped at 16000 output tokens (covers several
pages of dialogue comfortably). If a batch is big enough to blow through
that, the batch fails outright with a clear error rather than silently
translating only the first page or two and leaving the rest untranslated
with no explanation — that used to be exactly what happened before this
was checked for.

## Reading a finished batch

The "Read pages →" button opens a full-page reader: one page at a time,
in the batch's own page order (first to last — the order confirmed before
translating). Manga reads right-to-left, so *advancing* moves leftward:
the left arrow key (or the left on-screen button) steps to the next page,
the right arrow key/button steps back — the reverse of a left-to-right
reader (down/up follow the same advance/back sense as left/right,
respectively). The mouse wheel zooms toward the cursor, a zoomed-in page
can be dragged to pan, a "show original" toggle flips every page in the
reader between the original and the translation, and Escape (or clicking
outside the image) closes it.

## Debugging a bad translation

Every rendered page has a "debug info" link right next to it (also at
`/pages/{page_id}/debug`) with a JSON dump of every text block detected on
that page: its real bounding box, its estimated fill/text color, the raw
OCR text it was actually given to translate, the translation that came
back, and whether the model flagged that translation as a low-confidence
guess. If a rendered page looks wrong, this tells you which of a few very
different problems it actually is — a detection/merging miss, bad OCR, or
a translation mistake — directly, instead of having to guess from the
pixels.

## Correcting a translation by hand

A finished page has a "correct a line" section under it listing every
detected block's original OCR text alongside an editable field for its
translation, plus style controls: font size, background color, text color,
and the block's own position/size (x/y/width/height). The x/y/width/
height/font-size fields are pre-filled with whatever's *actually* in
effect right now — the automatic layout's real numbers, not a placeholder
that just says "auto" — so you can see exactly what you're nudging before
you touch it; only fields you actually change become a manual override
(saving without touching them leaves the automatic layout in charge, even
though the fields show real numbers). Editing and saving overwrites that
block's translation/style and re-renders that one page — it doesn't touch
detection, OCR, or re-run translation for anything else, so it's a fast,
free way to fix an occasional bad line (or nudge a block that landed
somewhere awkward) without waiting on (or paying for) the model again.
Hit "reset position/size" to go back to the automatic layout/font for
that block. See `edit_line` in `app/main.py`.

## Known limitations

- Each detected block gets its own flat-colored box, not the bubble's
  actual shape — on a round/oval bubble, this can look like a rectangular
  patch rather than filling the bubble's curve. This is a deliberate trade
  (see "Why it's built this way": a plain rectangle is simple and
  reliable; trying to reconstruct and fit into a bubble's actual outline
  is the harder, more fragile problem this app moved away from), not an
  oversight.
- The proximity-based merge that groups columns/lines into one block per
  bubble (see "Why it's built this way") is geometry, not real
  understanding of where one bubble ends and another begins — two
  genuinely separate bubbles positioned unusually close together could
  occasionally get merged into one translated block. Uncommon in practice
  (real bubbles have a visible gap between them), and a wrong merge is a
  content-quality issue (the model translates the merged text as one
  unit), not the illegible overlapping-text failure this replaced.
- The fill color for each block is estimated from what's immediately
  around it, not read from a certainty — a genuinely complex or gradient
  background can still get it wrong occasionally. A contrasting text
  outline is drawn as a backstop so the translation stays legible even
  when the color guess is off, rather than becoming invisible.
- If the model doesn't return a translation for every block it was given
  (it's asked to, but LLM instruction-following isn't guaranteed), that
  block is left as the untouched original with no indication in the UI —
  only a warning in the server log and a page's debug info. Not retried
  automatically.
- On a badly garbled OCR read, the model can still occasionally produce a
  plausible-sounding but wrong translation (hallucinating unrelated
  content) rather than a literal misread. Obvious model-authored
  meta-commentary leaking into the rendered text (a bracketed aside like
  "[unclear, ignore]") is filtered out defensively. The model is also
  asked to flag such a guess with a `low_confidence` field (logged
  server-side, visible in a page's debug info), but that's it
  self-reporting its own uncertainty, not a real check.
- The reading-order heuristic (right-to-left, top-to-bottom by row)
  assumes a roughly grid-like panel layout. Very irregular panel layouts
  aren't specifically handled — double-check page order in the UI before
  translating regardless, since it's shown to you either way.
- Single background worker, one batch at a time — fine for one person,
  not built for concurrent multi-user load.
- No accounts/auth — anyone who can reach the server can use it. Keep it
  off the open internet (LAN-only, or behind your own auth/reverse proxy)
  unless you add that yourself.
- Two local models run per page (text detector, OCR) — a few seconds a
  page on CPU in testing, faster with a GPU. Fine for a personal
  batch-at-a-time tool; not tuned for speed.

## Legal note

Manga is copyrighted. This tool doesn't publish or redistribute anything —
it renders translated pages back to you, in your own browser, and nothing
leaves your server except the OCR'd text and page images sent to your
configured provider's API for translation. Use it on scans you have the
right to read, and don't use it to redistribute translated copies to
others.
