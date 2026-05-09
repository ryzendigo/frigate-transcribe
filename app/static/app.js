(function () {
  "use strict";

  function onReady(fn) {
    if (document.readyState !== "loading") fn();
    else document.addEventListener("DOMContentLoaded", fn);
  }

  function isTypingTarget(el) {
    if (!el) return false;
    if (el.isContentEditable) return true;
    const tag = el.tagName;
    return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
  }

  function segmentRows() {
    return Array.from(document.querySelectorAll("#segments .seg-row"));
  }

  function currentFocusIndex(rows) {
    const el = document.querySelector(".seg-row.focus");
    if (!el) return -1;
    return rows.indexOf(el);
  }

  function setFocus(rows, i) {
    rows.forEach((r) => r.classList.remove("focus"));
    if (i < 0 || i >= rows.length) return;
    const target = rows[i];
    target.classList.add("focus");
    target.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  function moveFocus(delta) {
    const rows = segmentRows();
    if (!rows.length) return;
    const current = currentFocusIndex(rows);
    const next = Math.max(0, Math.min(rows.length - 1, current < 0 ? 0 : current + delta));
    setFocus(rows, next);
  }

  function scrollToLatest() {
    const rows = segmentRows();
    if (!rows.length) return;
    setFocus(rows, rows.length - 1);
  }

  function openEditForFocused() {
    const el = document.querySelector(".seg-row.focus");
    if (!el) return;
    const id = el.dataset.segId;
    if (!id) return;
    if (window.htmx) {
      htmx.ajax("GET", "/api/segments/" + id + "/edit", {
        target: "#segment-" + id,
        swap: "outerHTML",
      });
    } else {
      location.href = "/api/segments/" + id + "/edit";
    }
  }

  function navigateDay(delta) {
    const header = document.querySelector(".day-header[data-date]");
    if (!header) return;
    const target = delta < 0 ? header.dataset.prev : header.dataset.next;
    if (target) location.href = "/day/" + target;
  }

  function onKey(ev) {
    if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
    if (isTypingTarget(ev.target)) return;
    switch (ev.key) {
      case "/": {
        const s = document.getElementById("global-search");
        if (s) { ev.preventDefault(); s.focus(); s.select(); }
        return;
      }
      case "j": ev.preventDefault(); moveFocus(1); return;
      case "k": ev.preventDefault(); moveFocus(-1); return;
      case "n": ev.preventDefault(); navigateDay(1); return;
      case "p": ev.preventDefault(); navigateDay(-1); return;
      case "e": ev.preventDefault(); openEditForFocused(); return;
      case ".": ev.preventDefault(); scrollToLatest(); return;
    }
  }

  onReady(function () {
    document.addEventListener("keydown", onKey);
    // No auto-scroll on load — keeping the user's scroll position is more
    // important than parking at the latest segment. Press `.` to jump there.

    // Smooth-scroll any hash anchor to account for sticky header; if the exact
    // #time-HH-MM anchor isn't on the page (summary says 22:17 but the closest
    // conversation starts at 22:15), find the nearest conv-block or segment.
    if (location.hash) {
      setTimeout(function () {
        let el = document.querySelector(CSS.escape(location.hash.slice(1)));
        if (!el && /^#time-\d{2}-\d{2}$/.test(location.hash)) {
          const [hh, mm] = location.hash.slice(6).split("-").map(Number);
          const targetMinutes = hh * 60 + mm;
          const rows = segmentRows();
          let best = null;
          let bestDelta = Infinity;
          for (const r of rows) {
            const ts = r.querySelector("td.ts")?.textContent?.trim() || "";
            const m = ts.match(/^(\d{2}):(\d{2})/);
            if (!m) continue;
            const rowMins = Number(m[1]) * 60 + Number(m[2]);
            const delta = Math.abs(rowMins - targetMinutes);
            if (delta < bestDelta) { best = r; bestDelta = delta; }
          }
          el = best;
        }
        if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
      }, 80);
    }

    // Copy-to-clipboard buttons: data-copy="summary" grabs the neighbouring
    // <pre> or .summary-body. Graceful fallback for older browsers.
    document.querySelectorAll("button[data-copy]").forEach(function (btn) {
      btn.addEventListener("click", async function () {
        const scope = btn.closest(".summary-preview, .summary-body")?.parentElement
                   || btn.closest(".summary-preview, .summary-body")
                   || document;
        const src = scope.querySelector("pre, .summary-body");
        if (!src) return;
        const text = src.innerText.trim();
        try {
          await navigator.clipboard.writeText(text);
          const orig = btn.textContent;
          btn.textContent = "Copied";
          setTimeout(() => { btn.textContent = orig; }, 1500);
        } catch (err) {
          console.error("copy failed", err);
        }
      });
    });
  });
})();
