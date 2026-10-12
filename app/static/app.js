// No framework, no build step — a personal tool doesn't need one.

const body = document.body;
const batchId = body.dataset.batchId;
const batchStatus = body.dataset.batchStatus;

// --- Page reordering (draft state) ---------------------------------------

document.querySelectorAll(".move-up, .move-down").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const li = btn.closest("li[data-page-id]");
    const pageId = li.dataset.pageId;
    const direction = btn.classList.contains("move-up") ? "up" : "down";
    const res = await fetch(`/batches/${batchId}/move`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ page_id: pageId, direction }),
    });
    if (res.ok) {
      // Simplest correct thing: reload and let the server render fresh order.
      location.reload();
    }
  });
});

// --- Status polling while queued/processing -------------------------------

if (batchStatus === "queued" || batchStatus === "processing") {
  const poll = async () => {
    const res = await fetch(`/batches/${batchId}/status`);
    if (!res.ok) return;
    const data = await res.json();
    if (data.status !== batchStatus) {
      location.reload();
    }
  };
  setInterval(poll, 3000);
}

// --- Manual line correction (done state) -----------------------------------

function hexOf(input) {
  return input.value || null;
}

// The x/y/width/height/font-size fields are pre-filled with the current
// *automatic* layout (not left blank — see batch.html), so a plain click
// on Save without touching them must NOT lock that shown value in as a
// manual override. `data-auto` holds the baseline the field was
// pre-filled with; only a value that actually differs from it (or an
// explicitly emptied field) counts as the user overriding it.
function manualOrNull(input) {
  if (input.value === "" || input.value === input.dataset.auto) return null;
  return input.value;
}

async function saveLine(row) {
  const lineId = row.dataset.lineId;
  const pageId = row.dataset.pageId;
  const payload = {
    translated_text: row.querySelector(".correct-line-input").value,
    font_size: manualOrNull(row.querySelector(".style-font-size")),
    fill_color: hexOf(row.querySelector(".style-fill-color")),
    text_color: hexOf(row.querySelector(".style-text-color")),
    manual_x: manualOrNull(row.querySelector(".style-x")),
    manual_y: manualOrNull(row.querySelector(".style-y")),
    manual_w: manualOrNull(row.querySelector(".style-w")),
    manual_h: manualOrNull(row.querySelector(".style-h")),
  };
  const res = await fetch(`/lines/${lineId}/edit`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`status ${res.status}`);
  // Cache-bust so the freshly re-rendered page actually shows, not a
  // cached copy of the pre-edit version at the same URL.
  const img = document.querySelector(`.page-image[data-page-id="${pageId}"]`);
  if (img) img.src = `/pages/${pageId}/translated?t=${Date.now()}`;
}

document.querySelectorAll(".correct-line-save").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const row = btn.closest(".correct-line");
    btn.disabled = true;
    btn.textContent = "Saving…";
    try {
      await saveLine(row);
      btn.textContent = "Saved";
      setTimeout(() => { btn.textContent = "Save"; btn.disabled = false; }, 1200);
    } catch (err) {
      btn.textContent = "Failed — retry";
      btn.disabled = false;
    }
  });
});

document.querySelectorAll(".style-reset-box").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const row = btn.closest(".correct-line");
    // Setting each field back to its own data-auto baseline (rather than
    // blank) means manualOrNull sees "unchanged from auto" and sends null
    // for all of them — and the field still shows a real number
    // afterward, not an empty box.
    ["style-font-size", "style-x", "style-y", "style-w", "style-h"].forEach((cls) => {
      const input = row.querySelector(`.${cls}`);
      input.value = input.dataset.auto || "";
    });
    btn.disabled = true;
    btn.textContent = "Resetting…";
    try {
      await saveLine(row);
      btn.textContent = "reset position/size";
    } catch (err) {
      btn.textContent = "Failed — retry";
    }
    btn.disabled = false;
  });
});

// --- Original/translated toggle (done state) ------------------------------

document.querySelectorAll(".toggle-original").forEach((btn) => {
  btn.addEventListener("click", () => {
    const figure = btn.closest(".page-figure");
    const img = figure.querySelector(".toggle-img");
    const pageId = img.dataset.pageId;
    const showingOriginal = btn.dataset.showingOriginal === "true";
    if (showingOriginal) {
      img.src = `/pages/${pageId}/translated`;
      btn.textContent = "show original";
      btn.dataset.showingOriginal = "false";
    } else {
      img.src = `/pages/${pageId}/original`;
      btn.textContent = "show translated";
      btn.dataset.showingOriginal = "true";
    }
  });
});

// --- Full-page reader: sequential paging, arrow keys, wheel zoom ----------
//
// Manga reads right-to-left, so "advancing" moves leftward: the left
// arrow key/button steps to the next page (currentIndex + 1), the right
// arrow key/button steps back (currentIndex - 1) — the reverse of a
// left-to-right reader. The page order itself (pageIds) is unaffected by
// this: it's already first-to-last reading order regardless of which
// physical direction "advancing" points on screen.

const pagesGrid = document.querySelector(".pages-grid");
if (pagesGrid) {
  const pageIds = JSON.parse(pagesGrid.dataset.pages || "[]");
  const overlay = document.getElementById("reader-overlay");
  const readerImage = document.getElementById("reader-image");
  const readerCounter = document.getElementById("reader-counter");
  const leftBtn = document.getElementById("reader-left");   // advance (next page)
  const rightBtn = document.getElementById("reader-right"); // back (previous page)
  const toggleOriginalBtn = document.getElementById("reader-toggle-original");
  let currentIndex = 0;
  let zoom = 1;
  // Pixel offset applied (at the current zoom) on top of the image's own
  // centered position — how far it's been dragged. Both this and zoom are
  // folded into one `transform: translate() scale()` with a fixed
  // top-left origin (see applyTransform), rather than the CSS
  // transform-origin trick an earlier version used for cursor-anchored
  // zoom: that trick and free dragging don't compose cleanly (origin-based
  // zoom implicitly re-centers around a point, fighting any pan you've
  // already applied), while explicit translate+scale math handles both
  // together without surprises.
  let panX = 0;
  let panY = 0;
  // Applies across every page flipped to while the reader stays open, unlike
  // the per-page "show original" button in the grid below — one flip here
  // is meant to answer "let me compare the whole chapter", not just this page.
  let showingOriginal = false;
  // The image's own position with no pan/zoom applied — needed to convert
  // a screen coordinate (where the cursor is) into a position on the
  // untransformed image (see the wheel handler). Tracked ourselves rather
  // than re-derived from getBoundingClientRect() on every wheel event:
  // the CSS transition on `transform` means that during a fast scroll
  // (wheel events firing faster than the 50ms transition settles),
  // getBoundingClientRect() can return a still-animating, in-between
  // value, which fed back into the next step's math compounds into
  // visible drift — confirmed by instrumenting it, not assumed.
  let baseLeft = 0;
  let baseTop = 0;

  function captureBaseRect() {
    const r = readerImage.getBoundingClientRect();
    baseLeft = r.left - panX;
    baseTop = r.top - panY;
  }
  readerImage.addEventListener("load", captureBaseRect);

  function pageSrc(index) {
    return `/pages/${pageIds[index]}/${showingOriginal ? "original" : "translated"}`;
  }

  function applyTransform() {
    readerImage.style.transform = `translate(${panX}px, ${panY}px) scale(${zoom})`;
    readerImage.classList.toggle("zoomed", zoom > 1);
  }

  function showPage(index) {
    if (!pageIds.length || index < 0 || index >= pageIds.length) return;
    currentIndex = index;
    zoom = 1;
    panX = 0;
    panY = 0;
    applyTransform();
    // Best-effort immediately (still showing the previous image's box,
    // since the new one hasn't loaded yet) — the `load` listener above
    // corrects this once the new page's actual dimensions are known.
    captureBaseRect();
    readerImage.src = pageSrc(currentIndex);
    readerCounter.textContent = `${currentIndex + 1} / ${pageIds.length}`;
    // Dim and disable whichever direction has nowhere left to go, rather
    // than leaving both arrows looking equally clickable at either end.
    leftBtn.disabled = currentIndex === pageIds.length - 1;
    rightBtn.disabled = currentIndex === 0;
  }

  function openReader(index) {
    if (!pageIds.length) return;
    // Always start a fresh reading session on the translated version,
    // regardless of how a previous session was left.
    showingOriginal = false;
    toggleOriginalBtn.textContent = "show original";
    showPage(index);
    overlay.hidden = false;
  }

  function closeReader() {
    overlay.hidden = true;
  }

  // Only the explicit button opens the reader — page thumbnails in the
  // grid stay plain images (still usable with "show original"/the
  // correction fields right below them), not an implicit click-to-open.
  document.getElementById("reader-open").addEventListener("click", () => openReader(0));

  document.getElementById("reader-close").addEventListener("click", closeReader);
  leftBtn.addEventListener("click", () => showPage(currentIndex + 1));
  rightBtn.addEventListener("click", () => showPage(currentIndex - 1));
  toggleOriginalBtn.addEventListener("click", () => {
    showingOriginal = !showingOriginal;
    toggleOriginalBtn.textContent = showingOriginal ? "show translated" : "show original";
    // Swap just the image, not the whole page state — zoom/pan stays put so
    // a zoomed-in comparison between original and translated lines up.
    readerImage.src = pageSrc(currentIndex);
  });
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) closeReader();
  });

  document.addEventListener("keydown", (e) => {
    if (overlay.hidden) return;
    if (e.key === "Escape") closeReader();
    // Left/down = advance (next page), right/up = go back — see the
    // comment at the top of this block for why left means "next" here.
    else if (e.key === "ArrowLeft" || e.key === "ArrowDown") showPage(currentIndex + 1);
    else if (e.key === "ArrowRight" || e.key === "ArrowUp") showPage(currentIndex - 1);
  });

  // Mouse wheel zooms the current page in/out instead of scrolling the
  // (fullscreen, non-scrolling) overlay — clamped so it can't shrink past
  // fit-to-screen or balloon to an unusable size. Zooming toward the
  // cursor (rather than always the image's center) means the thing
  // you're actually pointing at is what gets bigger, not some fixed
  // point you then have to pan away from.
  //
  // The cursor's position in the image's own unscaled coordinate space is
  // (screenX - baseLeft - panX) / zoom (baseLeft/panX, not a fresh
  // getBoundingClientRect() — see where those are tracked, above), and
  // solving "keep that same local point under the cursor after zoom
  // changes" for the new pan gives the update below.
  readerImage.addEventListener("wheel", (e) => {
    e.preventDefault();
    const localX = (e.clientX - baseLeft - panX) / zoom;
    const localY = (e.clientY - baseTop - panY) / zoom;
    const newZoom = Math.min(5, Math.max(1, zoom - e.deltaY * 0.0015 * zoom));
    panX -= localX * (newZoom - zoom);
    panY -= localY * (newZoom - zoom);
    zoom = newZoom;
    if (zoom === 1) {
      // Fully zoomed out — snap back to centered rather than leaving it
      // wherever the last zoom-toward-cursor step happened to land it.
      panX = 0;
      panY = 0;
    }
    applyTransform();
  }, { passive: false });

  // Click-and-drag to pan around while zoomed in. Listening on `document`
  // for move/up (not just the image) means the drag keeps tracking even
  // if the cursor slides off the image or past the viewport edge while
  // the button's still held.
  let dragging = false;
  let dragStartX = 0;
  let dragStartY = 0;
  let dragPanX = 0;
  let dragPanY = 0;

  readerImage.addEventListener("mousedown", (e) => {
    if (zoom <= 1 || e.button !== 0) return;
    dragging = true;
    dragStartX = e.clientX;
    dragStartY = e.clientY;
    dragPanX = panX;
    dragPanY = panY;
    readerImage.classList.add("dragging");
    e.preventDefault(); // stop native image drag-ghost / text selection
  });

  document.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    panX = dragPanX + (e.clientX - dragStartX);
    panY = dragPanY + (e.clientY - dragStartY);
    applyTransform();
  });

  document.addEventListener("mouseup", () => {
    dragging = false;
    readerImage.classList.remove("dragging");
  });
}
