(() => {
  "use strict";

  const SVG_NS = "http://www.w3.org/2000/svg";
  const REFRESH_MS = 10_000;
  const STALE_MS = 25_000;
  const TIMELINE_PAGE = 18;
  const CATEGORY_ORDER = ["live", "memory", "work", "conversation", "tools"];
  const NODE_POSITIONS = [
    [205, 145], [410, 105], [655, 105], [850, 170], [930, 330],
    [820, 500], [625, 560], [440, 558], [255, 510], [130, 392],
    [125, 240], [310, 300], [905, 430], [690, 310], [350, 170],
  ];

  const elements = {
    chatLink: document.getElementById("chat-link"),
    clock: document.getElementById("observatory-clock"),
    connection: document.getElementById("observatory-connection"),
    status: document.getElementById("observatory-status"),
    refresh: document.getElementById("refresh-button"),
    liveToggle: document.getElementById("live-toggle"),
    liveToggleLabel: document.getElementById("live-toggle-label"),
    summary: document.getElementById("observatory-summary"),
    totalMemories: document.getElementById("total-memories"),
    totalContext: document.getElementById("total-context"),
    totalWork: document.getElementById("total-work"),
    totalSignals: document.getElementById("total-signals"),
    graphEdges: document.getElementById("graph-edges"),
    graphNodes: document.getElementById("graph-nodes"),
    graphLoading: document.getElementById("graph-loading"),
    graphVisibleCount: document.getElementById("graph-visible-count"),
    liveFreshness: document.getElementById("live-freshness"),
    stateCore: document.getElementById("state-core"),
    liveReadout: document.getElementById("live-readout"),
    inspectorTitle: document.getElementById("inspector-title"),
    inspectorDetail: document.getElementById("inspector-detail"),
    inspectorMeta: document.getElementById("inspector-meta"),
    timelineList: document.getElementById("timeline-list"),
    timelineMeta: document.getElementById("timeline-meta"),
    timelineEmpty: document.getElementById("timeline-empty"),
    timelineMore: document.getElementById("timeline-more"),
    memoryMeta: document.getElementById("memory-meta"),
    memoryStrengthFill: document.getElementById("memory-strength-fill"),
    memoryStrengthValue: document.getElementById("memory-strength-value"),
    memoryList: document.getElementById("memory-list"),
    memoryEmpty: document.getElementById("memory-empty"),
    workCurrent: document.getElementById("work-current"),
    workMeta: document.getElementById("work-meta"),
    archiveList: document.getElementById("archive-list"),
    workEmpty: document.getElementById("work-empty"),
    retentionList: document.getElementById("retention-list"),
    retentionCaveat: document.getElementById("retention-caveat"),
    error: document.getElementById("observatory-error"),
    errorMessage: document.getElementById("observatory-error-message"),
    retry: document.getElementById("retry-button"),
    announcer: document.getElementById("observatory-announcer"),
  };

  const query = new URLSearchParams(location.search);
  const initialRange = ["today", "7d", "retained"].includes(query.get("range"))
    ? query.get("range") : "today";
  const requestedFilters = new Set(
    String(query.get("signals") || CATEGORY_ORDER.join(",")).split(","),
  );
  const state = {
    token: "",
    snapshot: null,
    refreshing: false,
    live: true,
    range: initialRange,
    filters: new Set(CATEGORY_ORDER.filter(category => requestedFilters.has(category))),
    timelineLimit: TIMELINE_PAGE,
    selectedNode: "egg",
    controller: null,
    lastSuccessAt: 0,
    timer: null,
  };
  if (!state.filters.size) state.filters = new Set(CATEGORY_ORDER);

  function accessToken() {
    const fragment = new URLSearchParams(location.hash.replace(/^#/, ""));
    const supplied = fragment.get("access");
    if (supplied) sessionStorage.setItem("omni_access", supplied);
    return supplied || sessionStorage.getItem("omni_access") || "";
  }

  function authHeaders() {
    return { Authorization: `Bearer ${state.token}` };
  }

  function create(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function svgNode(tag, attributes = {}) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [name, value] of Object.entries(attributes)) {
      node.setAttribute(name, String(value));
    }
    return node;
  }

  function compactNumber(value) {
    const numeric = Number(value || 0);
    return new Intl.NumberFormat(undefined, {
      notation: numeric >= 1000 ? "compact" : "standard",
      maximumFractionDigits: 1,
    }).format(numeric);
  }

  function formatBytes(value) {
    let numeric = Number(value || 0);
    const units = ["B", "KB", "MB", "GB"];
    let unit = 0;
    while (numeric >= 1024 && unit < units.length - 1) {
      numeric /= 1024;
      unit += 1;
    }
    return `${numeric.toFixed(unit > 1 ? 1 : 0)} ${units[unit]}`;
  }

  function parsedDate(value) {
    if (!value) return null;
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? null : date;
  }

  function relativeTime(value) {
    const date = parsedDate(value);
    if (!date) return "time unknown";
    const seconds = Math.round((date.getTime() - Date.now()) / 1000);
    const magnitude = Math.abs(seconds);
    const formatter = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });
    if (magnitude < 60) return formatter.format(seconds, "second");
    if (magnitude < 3600) return formatter.format(Math.round(seconds / 60), "minute");
    if (magnitude < 86400) return formatter.format(Math.round(seconds / 3600), "hour");
    return formatter.format(Math.round(seconds / 86400), "day");
  }

  function shortTime(value) {
    const date = parsedDate(value);
    if (!date) return "—";
    if (state.range === "today") {
      return new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit" }).format(date);
    }
    return new Intl.DateTimeFormat(undefined, {
      month: "short", day: "numeric", hour: "numeric", minute: "2-digit",
    }).format(date);
  }

  function inRange(value) {
    if (state.range === "retained") return true;
    const date = parsedDate(value);
    if (!date) return false;
    const now = new Date();
    if (state.range === "7d") return now.getTime() - date.getTime() <= 7 * 86400 * 1000;
    return date.getFullYear() === now.getFullYear()
      && date.getMonth() === now.getMonth()
      && date.getDate() === now.getDate();
  }

  function updateUrlState() {
    const parameters = new URLSearchParams(location.search);
    parameters.set("range", state.range);
    parameters.set("signals", CATEGORY_ORDER.filter(category => state.filters.has(category)).join(","));
    const suffix = parameters.toString();
    history.replaceState(null, "", `${location.pathname}${suffix ? `?${suffix}` : ""}${location.hash}`);
  }

  function setConnection(mode, text) {
    elements.connection.classList.remove("online", "stale", "offline");
    elements.connection.classList.add(mode);
    elements.status.textContent = text;
  }

  function serviceSummary(services) {
    const entries = Object.entries(services || {});
    const healthy = entries.filter(([, value]) => value && value.ok === true).length;
    return { healthy, total: entries.length };
  }

  function timelineCounts(snapshot) {
    const counts = { conversation: 0, tools: 0, work: 0, memory: 0, live: 0 };
    for (const item of snapshot.timeline || []) {
      if (counts[item.category] !== undefined) counts[item.category] += 1;
    }
    return counts;
  }

  function graphNodes(snapshot) {
    const live = snapshot.live || {};
    const memory = snapshot.memory || {};
    const passive = memory.passive || {};
    const virtual = memory.virtual_context || {};
    const virtualStorage = memory.virtual_storage || {};
    const work = snapshot.work || {};
    const archive = work.archive || {};
    const environment = live.environment || {};
    const battery = environment.battery || {};
    const runtimeMemory = environment.memory || {};
    const connectivity = environment.connectivity || {};
    const requests = live.requests || {};
    const daemon = live.daemon || {};
    const harness = live.harness || {};
    const services = serviceSummary(live.services);
    const counts = timelineCounts(snapshot);
    const nodes = [
      {
        id: "egg", label: "EGG", value: harness.state || daemon.state || "present",
        category: "root", detail: "The joined view of evidence Egg currently retains.",
        source: "observatory projection", retention: "mixed · inspect each connected node",
        freshness: snapshot.captured_at, status: "ok", x: 540, y: 325, radius: 57,
      },
      {
        id: "presence", label: "VOICE LOOP", value: harness.state || "unavailable",
        category: "live", detail: "Current ReSpeaker / call-harness interaction state.",
        source: "harness-status.json", retention: "live sample", freshness: harness.updated_at,
        status: !harness.available ? "error" : (harness.fresh ? "ok" : "stale"),
      },
      {
        id: "services", label: "SERVICES", value: `${services.healthy}/${services.total} healthy`,
        category: "live", detail: "Direct health probes for the adapter, comprehension, speech, and language services.",
        source: "local health endpoints", retention: "live sample", freshness: snapshot.captured_at,
        status: services.healthy === services.total ? "ok" : "error",
      },
      {
        id: "compute", label: "COMPUTE", value: `${Number(runtimeMemory.used_percent || 0).toFixed(0)}% memory`,
        category: "live", detail: `Host load and bounded resource facts. ${Number(requests.active || 0)} active inference request(s).`,
        source: "bounded runtime snapshot", retention: "5-second sampled cache", freshness: environment.captured_at,
        status: "ok",
      },
      {
        id: "battery", label: "BATTERY", value: battery.available ? `${battery.percentage}%` : "unavailable",
        category: "live", detail: battery.available
          ? `${battery.voltage_v} V${battery.charging === true ? " · charging" : ""}`
          : String(battery.reason || "Battery service has no fresh reading."),
        source: battery.source || "EGG battery service", retention: "live bounded state", freshness: environment.captured_at,
        status: battery.available ? "ok" : "stale",
      },
      {
        id: "network", label: "NETWORK", value: connectivity.active_link ? "attached" : "unavailable",
        category: "live", detail: connectivity.note || "Local attachment only; public reachability is not assumed.",
        source: "host route and link state", retention: "live sample", freshness: environment.captured_at,
        status: connectivity.active_link ? "ok" : "stale",
      },
      {
        id: "passive-memory", label: "MEMORY", value: `${compactNumber(passive.entries)} retained`,
        category: "memory", detail: "Semantic voice memories strengthen when recalled and decay when repeatedly unused.",
        source: "passive memory SQLite", retention: "durable · relevance-decayed", freshness: passive.newest_at,
        status: passive.available ? "ok" : "stale",
      },
      {
        id: "working-context", label: "CONTEXT", value: `${compactNumber(virtual.chunks)} chunks`,
        category: "memory", detail: `${compactNumber(virtualStorage.corpora)} retained session corpora; the current physical model window remains bounded.`,
        source: "virtual-context evidence stores", retention: "durable until explicit session clear", freshness: snapshot.captured_at,
        status: virtual.mode === "off" ? "stale" : "ok",
      },
      {
        id: "session-memory", label: "SESSION NOTES", value: `${compactNumber(memory.session?.entries)} entries`,
        category: "memory", detail: "Explicit working notes scoped to this authenticated browser session.",
        source: "portal session memory", retention: "short-lived session state", freshness: snapshot.captured_at,
        status: "ok",
      },
      {
        id: "current-work", label: "CURRENT WORK", value: `${compactNumber(work.live?.length)} active`,
        category: "work", detail: "Background objectives, progress checkpoints, tools, and verified outcomes.",
        source: "background task store", retention: "durable while active", freshness: work.live?.[0]?.updated_at,
        status: work.live?.some(item => item.status === "blocked") ? "error" : "ok",
      },
      {
        id: "work-archive", label: "OUTCOMES", value: `${compactNumber(archive.records_observed)} observed`,
        category: "work", detail: archive.coverage_note || "Human-readable archive of completed, blocked, and cancelled objectives.",
        source: "background task archive", retention: "durable archive", freshness: archive.recent?.[0]?.updated_at,
        status: archive.available ? "ok" : "stale",
      },
      {
        id: "conversation", label: "CONVERSATION", value: `${compactNumber(counts.conversation)} signals`,
        category: "conversation", detail: "Content-redacted request stage evidence for this browser session.",
        source: "session diagnostic journal", retention: "short-lived session journal", freshness: snapshot.timeline?.[0]?.at,
        status: "ok",
      },
      {
        id: "tools", label: "TOOLS", value: `${compactNumber(counts.tools)} calls`,
        category: "tools", detail: "Tool name, round, and outcome only; arguments and results are not copied here.",
        source: "session diagnostic journal", retention: "short-lived session journal", freshness: snapshot.timeline?.find(item => item.category === "tools")?.at,
        status: snapshot.timeline?.some(item => item.category === "tools" && item.status === "error") ? "error" : "ok",
      },
    ];
    if (snapshot.location?.available === true) {
      const location = snapshot.location;
      nodes.push({
        id: "location", label: "LOCATION", value: [location.city, location.region].filter(Boolean).join(", ") || "approximate",
        category: "live", detail: "Approximate browser network-area evidence; never treated as GPS, street, or visual evidence.",
        source: "session-scoped IP geolocation", retention: "short-lived session state", freshness: snapshot.captured_at,
        status: "ok",
      });
    }
    let position = 0;
    for (const node of nodes) {
      if (node.id === "egg") continue;
      [node.x, node.y] = NODE_POSITIONS[position] || [540, 325];
      node.radius = node.category === "memory" || node.category === "work" ? 38 : 34;
      position += 1;
    }
    return nodes;
  }

  function visibleGraphNodes(snapshot) {
    return graphNodes(snapshot).filter(node => node.id === "egg" || state.filters.has(node.category));
  }

  function drawGraph(snapshot) {
    elements.graphEdges.replaceChildren();
    elements.graphNodes.replaceChildren();
    const nodes = visibleGraphNodes(snapshot);
    const root = nodes.find(node => node.id === "egg");
    for (const node of nodes) {
      if (node.id === "egg") continue;
      const line = svgNode("line", { x1: root.x, y1: root.y, x2: node.x, y2: node.y });
      line.classList.add("graph-edge");
      if (node.status === "error") line.classList.add("error");
      else if (node.category === "live") line.classList.add("active");
      else line.classList.add("retained");
      elements.graphEdges.appendChild(line);
    }
    for (const node of nodes) {
      const group = svgNode("g", {
        transform: `translate(${node.x} ${node.y})`, tabindex: "0", role: "button",
        "aria-label": `${node.label}: ${node.value}`,
      });
      group.classList.add("graph-node", `category-${node.category}`, `status-${node.status}`);
      if (node.id === "egg") group.classList.add("root-node");
      if (node.id === state.selectedNode) group.classList.add("is-selected");
      const title = svgNode("title");
      title.textContent = `${node.label} — ${node.detail}`;
      const ring = svgNode("circle", { class: "node-ring", r: node.radius });
      const core = svgNode("circle", { class: "node-core", r: node.id === "egg" ? 8 : 4 });
      const label = svgNode("text", { class: "node-label", y: node.radius + 19 });
      label.textContent = node.label;
      const value = svgNode("text", { class: "node-value", y: node.radius + 33 });
      value.textContent = node.value;
      group.append(title, ring, core, label, value);
      const select = () => {
        state.selectedNode = node.id;
        drawGraph(snapshot);
        renderInspector(node);
      };
      group.addEventListener("click", select);
      group.addEventListener("keydown", event => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          select();
        }
      });
      elements.graphNodes.appendChild(group);
    }
    const selected = nodes.find(node => node.id === state.selectedNode) || root;
    if (selected.id !== state.selectedNode) state.selectedNode = selected.id;
    renderInspector(selected);
    elements.graphVisibleCount.textContent = `${nodes.length} visible node${nodes.length === 1 ? "" : "s"}`;
  }

  function renderInspector(node) {
    elements.inspectorTitle.textContent = node.label;
    elements.inspectorDetail.textContent = node.detail;
    elements.inspectorMeta.replaceChildren();
    const rows = [
      ["Value", node.value],
      ["Source", node.source],
      ["Retention", node.retention],
      ["Freshness", node.freshness ? relativeTime(node.freshness) : "not observed"],
    ];
    for (const [label, value] of rows) {
      const wrapper = create("div");
      wrapper.append(create("dt", "", label), create("dd", "", value));
      elements.inspectorMeta.appendChild(wrapper);
    }
  }

  function renderLive(snapshot) {
    const live = snapshot.live || {};
    const daemon = live.daemon || {};
    const harness = live.harness || {};
    const environment = live.environment || {};
    const battery = environment.battery || {};
    const runtimeMemory = environment.memory || {};
    const requests = live.requests || {};
    const services = serviceSummary(live.services);
    const age = Math.max(0, (Date.now() - new Date(snapshot.captured_at).getTime()) / 1000);
    const fresh = age < STALE_MS / 1000;
    elements.liveFreshness.textContent = fresh ? "Fresh sample" : "Stale sample";
    elements.liveFreshness.className = `freshness ${fresh ? "fresh" : "stale"}`;
    elements.stateCore.className = "state-core";
    if (!daemon.available) elements.stateCore.classList.add("error");
    else if (["thinking", "speaking", "hearing"].includes(harness.state)) elements.stateCore.classList.add("thinking");
    else elements.stateCore.classList.add("online");
    const rows = [
      ["Presence", harness.state || "unavailable"],
      ["Runtime", daemon.state || "unavailable"],
      ["Services", `${services.healthy}/${services.total} healthy`],
      ["Inference", `${Number(requests.active || 0)} active · ${Number(requests.queued || 0)} queued`],
      ["Memory", `${Number(runtimeMemory.used_percent || 0).toFixed(1)}% host RAM`],
      ["Battery", battery.available ? `${battery.percentage}% · ${battery.voltage_v} V` : "unavailable"],
      ["Model", daemon.model || document.body.dataset.model || "unknown"],
    ];
    elements.liveReadout.replaceChildren();
    for (const [label, value] of rows) {
      const wrapper = create("div");
      wrapper.append(create("dt", "", label), create("dd", "", value));
      elements.liveReadout.appendChild(wrapper);
    }
  }

  function filteredTimeline(snapshot) {
    return (snapshot.timeline || []).filter(item => state.filters.has(item.category) && inRange(item.at));
  }

  function renderTimeline(snapshot) {
    const filtered = filteredTimeline(snapshot);
    const shown = filtered.slice(0, state.timelineLimit);
    elements.timelineList.replaceChildren();
    for (const item of shown) {
      const row = create("li", "timeline-item");
      const time = create("time", "timeline-time", shortTime(item.at));
      if (item.at) time.dateTime = item.at;
      const marker = create("span", `timeline-marker ${item.category}`);
      marker.setAttribute("aria-hidden", "true");
      const body = create("div", "timeline-body");
      body.append(create("strong", "", item.label || "signal"));
      if (item.detail) body.append(create("p", "", item.detail));
      const source = create("span", "timeline-source", item.source || "retained evidence");
      row.append(time, marker, body, source);
      elements.timelineList.appendChild(row);
    }
    elements.timelineEmpty.hidden = filtered.length > 0;
    elements.timelineMore.hidden = shown.length >= filtered.length;
    elements.timelineMore.textContent = `Show ${Math.min(TIMELINE_PAGE, filtered.length - shown.length)} more retained signals`;
    const rangeLabels = { today: "today", "7d": "the last 7 days", retained: "all retained time" };
    elements.timelineMeta.textContent = `${filtered.length} signals across ${rangeLabels[state.range]}`;
  }

  function renderMemory(snapshot) {
    const passive = snapshot.memory?.passive || {};
    const memories = (passive.recent || []).filter(item => inRange(item.created_at));
    const strength = Math.max(0, Math.min(1, Number(passive.mean_strength || 0)));
    elements.memoryStrengthFill.style.width = `${strength * 100}%`;
    elements.memoryStrengthValue.textContent = strength ? strength.toFixed(3) : "—";
    elements.memoryMeta.textContent = passive.available
      ? `${compactNumber(passive.entries)} durable · oldest ${Number(passive.oldest_days || 0).toFixed(1)} days`
      : "Passive memory unavailable";
    elements.memoryList.replaceChildren();
    for (const item of memories.slice(0, 16)) {
      const row = create("li", "memory-item");
      const text = create("p", "", item.text || "Retained memory");
      const kind = create("span", "memory-kind", item.kind || "turn");
      const time = create("time", "", shortTime(item.created_at));
      if (item.created_at) time.dateTime = item.created_at;
      const vitals = create("div", "memory-vitals");
      vitals.append(
        create("span", "", `strength ${Number(item.strength || 0).toFixed(3)}`),
        create("span", "", `${Number(item.uses || 0)} recall${Number(item.uses || 0) === 1 ? "" : "s"}`),
      );
      row.append(text, kind, time, vitals);
      elements.memoryList.appendChild(row);
    }
    elements.memoryEmpty.hidden = memories.length > 0;
  }

  function renderWork(snapshot) {
    const work = snapshot.work || {};
    const live = work.live || [];
    const archive = work.archive || {};
    elements.workCurrent.replaceChildren();
    for (const task of live) {
      const card = create("article", "work-card");
      const header = create("header");
      header.append(
        create("span", "work-status", String(task.status || "active").replaceAll("_", " ")),
        create("time", "", relativeTime(task.updated_at ? new Date(Number(task.updated_at) * 1000).toISOString() : null)),
      );
      card.append(header, create("h3", "", task.objective || "Background objective"));
      if (task.current_stage) card.append(create("p", "", task.current_stage));
      const progress = create("div", "work-progress");
      progress.append(
        create("span", "", `round ${Number(task.round || 0)}`),
        create("span", "", `${Number(task.actions?.length || 0)} actions`),
        create("span", "", `${Number(task.tools_used?.length || 0)} tools`),
      );
      card.append(progress);
      elements.workCurrent.appendChild(card);
    }
    elements.archiveList.replaceChildren();
    const archived = (archive.recent || []).filter(item => inRange(item.updated_at || item.created_at));
    for (const item of archived.slice(0, 12)) {
      const row = create("li", "archive-item");
      const time = create("time", "", shortTime(item.updated_at || item.created_at));
      row.append(
        create("span", "", item.status || "archived"),
        create("p", "", item.objective || "Archived objective"),
        time,
      );
      elements.archiveList.appendChild(row);
    }
    elements.workEmpty.hidden = live.length > 0 || archived.length > 0;
    elements.workMeta.textContent = `${live.length} active · ${archive.records_observed || 0} archived records observed`;
  }

  function renderRetention(snapshot) {
    elements.retentionList.replaceChildren();
    for (const item of snapshot.retention || []) {
      const row = create("li", "retention-item");
      const body = create("div");
      body.append(
        create("strong", "", item.label),
        create("p", "", item.detail),
        create("span", "", item.scope),
      );
      row.appendChild(body);
      elements.retentionList.appendChild(row);
    }
    elements.retentionCaveat.textContent = snapshot.scope?.caveat || "";
  }

  function renderSummary(snapshot) {
    const passive = snapshot.memory?.passive || {};
    const virtual = snapshot.memory?.virtual_context || {};
    const work = snapshot.work || {};
    const filtered = filteredTimeline(snapshot);
    const active = Number(work.live?.length || 0);
    const voice = snapshot.live?.harness?.state || "unavailable";
    elements.totalMemories.textContent = compactNumber(passive.entries);
    elements.totalContext.textContent = compactNumber(virtual.chunks);
    elements.totalWork.textContent = compactNumber(active);
    elements.totalSignals.textContent = compactNumber(filtered.length);
    elements.summary.textContent = `Egg is ${voice}. It retains ${compactNumber(passive.entries)} semantic memories, ${compactNumber(virtual.chunks)} current-session context chunks, and ${active} active objective${active === 1 ? "" : "s"}.`;
  }

  function render(snapshot) {
    renderSummary(snapshot);
    drawGraph(snapshot);
    renderLive(snapshot);
    renderTimeline(snapshot);
    renderMemory(snapshot);
    renderWork(snapshot);
    renderRetention(snapshot);
    elements.graphLoading.hidden = true;
  }

  function showError(error) {
    const message = error instanceof Error ? error.message : String(error);
    elements.errorMessage.textContent = message;
    elements.error.hidden = false;
    setConnection(state.snapshot ? "stale" : "offline", state.snapshot ? "Stale" : "Offline");
    if (!state.snapshot) elements.graphLoading.querySelector("p").textContent = "Local evidence unavailable";
  }

  async function refresh({ announce = false } = {}) {
    if (state.refreshing) return;
    if (!state.token) {
      showError(new Error("This link is missing its access fragment. Return to chat and open the Observatory again."));
      return;
    }
    state.refreshing = true;
    elements.refresh.disabled = true;
    elements.refresh.classList.add("refreshing");
    if (state.controller) state.controller.abort();
    const controller = new AbortController();
    state.controller = controller;
    try {
      const response = await fetch("/api/observatory", {
        headers: authHeaders(),
        signal: controller.signal,
      });
      if (!response.ok) {
        let detail = `Observatory returned HTTP ${response.status}`;
        try {
          const payload = await response.json();
          if (payload.error) detail = payload.error;
        } catch (_error) {
          // Preserve the bounded HTTP status when a proxy returned non-JSON.
        }
        throw new Error(detail);
      }
      const snapshot = await response.json();
      if (snapshot.schema !== "robit.omni.observatory.v1") {
        throw new Error("Observatory returned an unsupported snapshot schema");
      }
      state.snapshot = snapshot;
      state.lastSuccessAt = Date.now();
      elements.error.hidden = true;
      setConnection("online", "Live");
      render(snapshot);
      if (announce) elements.announcer.textContent = `Observatory refreshed ${relativeTime(snapshot.captured_at)}.`;
    } catch (error) {
      if (error.name !== "AbortError") showError(error);
    } finally {
      if (state.controller === controller) state.controller = null;
      state.refreshing = false;
      elements.refresh.disabled = false;
      elements.refresh.classList.remove("refreshing");
    }
  }

  function applyControls() {
    document.querySelectorAll("[data-range]").forEach(button => {
      button.setAttribute("aria-pressed", String(button.dataset.range === state.range));
      button.addEventListener("click", () => {
        state.range = button.dataset.range;
        state.timelineLimit = TIMELINE_PAGE;
        document.querySelectorAll("[data-range]").forEach(item => {
          item.setAttribute("aria-pressed", String(item === button));
        });
        updateUrlState();
        if (state.snapshot) render(state.snapshot);
      });
    });
    document.querySelectorAll("[data-filter]").forEach(button => {
      button.setAttribute("aria-pressed", String(state.filters.has(button.dataset.filter)));
      button.addEventListener("click", () => {
        const category = button.dataset.filter;
        if (state.filters.has(category) && state.filters.size > 1) state.filters.delete(category);
        else state.filters.add(category);
        button.setAttribute("aria-pressed", String(state.filters.has(category)));
        state.timelineLimit = TIMELINE_PAGE;
        updateUrlState();
        if (state.snapshot) render(state.snapshot);
      });
    });
  }

  function updateClock() {
    elements.clock.textContent = new Intl.DateTimeFormat(undefined, {
      weekday: "short", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit",
    }).format(new Date());
    if (state.lastSuccessAt && Date.now() - state.lastSuccessAt > STALE_MS) {
      setConnection("stale", "Stale");
      elements.liveFreshness.textContent = `Last sample ${relativeTime(state.lastSuccessAt)}`;
      elements.liveFreshness.className = "freshness stale";
    }
  }

  function initialize() {
    state.token = accessToken();
    if (location.hash) elements.chatLink.href = `/${location.hash}`;
    applyControls();
    elements.refresh.addEventListener("click", () => refresh({ announce: true }));
    elements.retry.addEventListener("click", () => refresh({ announce: true }));
    elements.timelineMore.addEventListener("click", () => {
      state.timelineLimit += TIMELINE_PAGE;
      if (state.snapshot) renderTimeline(state.snapshot);
    });
    elements.liveToggle.addEventListener("click", () => {
      state.live = !state.live;
      elements.liveToggle.setAttribute("aria-pressed", String(state.live));
      elements.liveToggleLabel.textContent = state.live ? "Live refresh" : "Refresh paused";
      elements.announcer.textContent = state.live ? "Live refresh resumed." : "Live refresh paused.";
      if (state.live) refresh();
    });
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden && state.live) refresh();
    });
    updateClock();
    setInterval(updateClock, 1000);
    state.timer = setInterval(() => {
      if (state.live && !document.hidden) refresh();
    }, REFRESH_MS);
    refresh();
  }

  initialize();
})();
