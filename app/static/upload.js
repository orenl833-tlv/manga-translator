// Lets the upload form accept a pasted image (Ctrl+V) in addition to the
// normal file picker — merges into the same <input type="file"> so both
// paths feed the same form submission.

(function () {
  const input = document.getElementById("file-input");
  const chipsEl = document.getElementById("file-chips");
  if (!input || !chipsEl) return; // not on this page

  function renderChips() {
    chipsEl.innerHTML = "";
    Array.from(input.files).forEach((file) => {
      const chip = document.createElement("span");
      chip.className = "file-chip";
      chip.textContent = file.name;
      chipsEl.appendChild(chip);
    });
  }

  function addFiles(newFiles) {
    // Native file inputs don't support appending programmatically except
    // via DataTransfer — rebuild the FileList as existing + new.
    const dt = new DataTransfer();
    Array.from(input.files).forEach((file) => dt.items.add(file));
    newFiles.forEach((file) => dt.items.add(file));
    input.files = dt.files;
    renderChips();
  }

  input.addEventListener("change", renderChips);

  document.addEventListener("paste", (event) => {
    const items = event.clipboardData ? event.clipboardData.items : [];
    const imageFiles = [];
    let counter = 0;
    for (const item of items) {
      if (item.type && item.type.startsWith("image/")) {
        const file = item.getAsFile();
        if (!file) continue;
        counter += 1;
        // Pasted images usually arrive with a generic/empty filename —
        // give each a distinct one (natural_sort_key downstream needs
        // something to sort by, and duplicate names would just overwrite).
        const ext = item.type.split("/")[1] || "png";
        imageFiles.push(new File([file], `pasted-${Date.now()}-${counter}.${ext}`, { type: file.type }));
      }
    }
    if (imageFiles.length) {
      event.preventDefault();
      addFiles(imageFiles);
    }
  });

  renderChips();
})();
