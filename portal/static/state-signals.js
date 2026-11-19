/* Source-time tracks for authorized input, processing, tools, and replies. */
(() => {
  "use strict";
  const instrument = document.getElementById("state-signal-lanes");
  if (!instrument) return;
  const lanes = [["heard", "Heard"], ["thought", "Thinking"], ["done", "Tools"], ["said", "Said"]];
  const windows = { 60: 15, 300: 60, 900: 180 };
  const scope = document.getElementById("state-signal-scope");
  let windowSeconds = 300;
  let sample = null;
  let selectedId = null;
  let fingerprint = "";
  let markers = [];
  let ruler = null;
  let detailTitle;
  let detailText;
  let detailOrigin;

  function node(tag, cls, text) {
    const element = document.createElement(tag);
    element.className = cls;
    if (text !== undefined) element.textContent = text;
    return element;
  }

  function selected(item) {
    selectedId = item?.id ?? null;
    for (const {button, value} of markers) {
      button.setAttribute("aria-pressed", String(value.id === selectedId));
    }
    detailTitle.textContent = item ? `${item.label} · ${new Date(item.at * 1000).toLocaleTimeString()}` : "Select an event to inspect its source.";
    detailText.textContent = item?.text || "";
    detailOrigin.textContent = item ? [item.authority, item.actor, item.request_id && `Request ${item.request_id}`,
      item.source_sha256 && `SHA-256 ${item.source_sha256}`,
      item.audio_generated && "Audio generated · playback unverified",
      item.text_truncated && "Excerpt truncated"].filter(Boolean).join(" · ") : "";
  }

  function placeMarkers(animate = false) {
    if (!ruler) return;
    const now = Date.now() / 1000;
    const width = ruler.clientWidth;
    if (!width) return;
    const previous = new Map();
    const labelWidth = Math.min(92, Math.max(52, width * 0.16));
    for (const {button, value} of markers) {
      const elapsed = now - value.at;
      const x = width * (1 - elapsed / windowSeconds);
      button.hidden = x < 0 || x > width + 3;
      if (button.hidden) continue;
      const next = previous.get(value.lane) || [-Infinity, -Infinity];
      const stack = next[0] + labelWidth <= x ? 0 : 1;
      const compact = next[stack] + labelWidth > x;
      next[stack] = x;
      previous.set(value.lane, next);
      button.classList.toggle("is-compact", compact);
      button.classList.toggle("is-moving", animate);
      button.style.setProperty("--signal-x", `${Math.min(width, Math.max(7, x))}px`);
      button.style.setProperty("--signal-stack", stack);
      button.style.setProperty("--signal-width", `${labelWidth}px`);
    }
  }

  function draw() {
    if (!sample) return;
    const now = Date.now() / 1000;
    const sources = sample.entries.filter(item => item.at >= now - windowSeconds && item.at <= now + 1);
    scope.textContent = `${sample.scope} · ${sources.length} recorded events`;
    const key = JSON.stringify([windowSeconds, sample.scope, sources]);
    if (key === fingerprint) {
      placeMarkers();
      return;
    }
    fingerprint = key;
    const focusId = document.activeElement?.dataset.signalId;
    const chart = node("div", "signal-chart");
    markers = [];
    for (const [lane, label] of lanes) {
      const row = node("div", "signal-track");
      const track = node("div", "signal-strip");
      track.setAttribute("role", "group");
      track.setAttribute("aria-label", label);
      for (const item of sources.filter(record => record.lane === lane).sort((a, b) => a.at - b.at)) {
        const button = node("button", "signal-stamp", item.text.replace(/\s+/g, " ").trim() || item.label);
        button.type = "button";
        button.dataset.signalId = item.id;
        button.title = `${item.label} · ${new Date(item.at * 1000).toLocaleTimeString()}`;
        button.setAttribute("aria-label", button.title);
        button.addEventListener("click", () => selected(item));
        markers.push({button, value: item});
        track.append(button);
      }
      row.append(node("span", "signal-lane-name", label), track);
      chart.append(row);
    }
    const axis = node("div", "signal-track signal-axis-row");
    ruler = node("div", "signal-axis-labels");
    const step = windows[windowSeconds];
    for (let elapsed = windowSeconds; elapsed >= 0; elapsed -= step) {
      const label = elapsed === 0 ? "Now" : windowSeconds === 60 ? `−${elapsed}s` : `−${elapsed / 60}m`;
      ruler.append(node("span", "", label));
    }
    chart.style.setProperty("--minor-tick", `${step / windowSeconds * 25}%`);
    axis.append(node("span", "", ""), ruler);
    chart.append(axis);
    if (!sources.length) chart.append(node("p", "signal-source-empty", "No recorded events in this window."));
    const detail = node("section", "signal-inspection");
    detailTitle = node("h4", "signal-source-heading");
    detailText = node("p", "signal-source-text");
    detailText.tabIndex = 0;
    detailOrigin = node("p", "signal-source-authority");
    detail.setAttribute("aria-live", "polite");
    detail.append(detailTitle, detailText, detailOrigin);
    instrument.replaceChildren(chart, detail);
    selected(sources.find(item => item.id === selectedId));
    placeMarkers();
    if (focusId) markers.find(({value}) => value.id === focusId)?.button.focus({preventScroll: true});
  }

  function render(value, processing = []) {
    if (!value) return;
    const stages = processing.map(item => ({
      ...item, at: Date.parse(item.at) / 1000, lane: "thought",
      label: "Processing", text: item.label, authority: "Recorded inference stage",
    }));
    sample = {scope: value.scope || "This browser", entries: [...(value.sources || []), ...stages]};
    draw();
  }

  const observer = new ResizeObserver(() => placeMarkers());
  observer.observe(instrument);
  // CSS interpolates between clock samples. Reduced-motion clients still
  // receive accurate positions whenever a snapshot renders.
  const motion = matchMedia("(prefers-reduced-motion: reduce)");
  setInterval(() => {
    if (!document.hidden && instrument.offsetWidth && !motion.matches) placeMarkers(true);
  }, 250);
  window.OmniStateSignals = {
    render,
    setWindow(seconds) {
      windowSeconds = seconds;
      draw();
    },
    clear() {
      sample = null;
      fingerprint = "";
      selectedId = null;
      markers = [];
      ruler = null;
      instrument.replaceChildren();
    },
  };
})();
