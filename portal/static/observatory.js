/* LCD display modules and connected readouts adapted from Nick Gallo:
   https://www.nickgallo.dev/projects/clock/ — see docs/observatory-lcd.md. */
(() => {
  "use strict";

  const $ = id => document.getElementById(id);
  const REFRESH_MS = 2_000;
  const STALE_MS = 25_000;
  const PAGE_SIZE = 18;
  const TABS = ["weights", "events", "states", "memory"];
  const query = new URLSearchParams(location.search);
  const state = {
    token: "", snapshot: null, refreshing: false, live: true,
    tab: TABS.includes(query.get("tab")) ? query.get("tab") : "weights",
    range: ["today", "7d", "retained"].includes(query.get("range")) ? query.get("range") : "today",
    filter: ["all", "conversation", "tools", "hearing", "llm", "speaking", "vision"].includes(query.get("events")) ? query.get("events") : "all",
    limit: PAGE_SIZE, controller: null, lastSuccessAt: 0,
    agent: null, savingIdentity: false, identityUpdatedAt: 0, window: 60, selectedEventId: null,
    chatActivity: null, chatObservedAt: 0,
  };
  const pad = (value, width = 2) => String(value).padStart(width, "0");
  const compact = value => new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 }).format(value);
  const numeric = value => value !== null && value !== undefined && value !== "" && Number.isFinite(Number(value));
  const count = value => numeric(value) ? compact(Number(value)) : "unreported";
  const clamp = value => Math.max(0, Math.min(100, Number(value)));

  function create(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function parsedDate(value) {
    if (!value) return null;
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? null : date;
  }

  function timestamp(value) {
    const date = parsedDate(value);
    if (!date) return "Time unreported";
    return new Intl.DateTimeFormat(undefined, {
      month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit", hourCycle: "h23",
    }).format(date);
  }

  function inRange(value) {
    if (state.range === "retained") return true;
    const date = parsedDate(value);
    if (!date) return false;
    const now = new Date();
    if (state.range === "7d") return date <= now && now - date <= 7 * 86400 * 1000;
    return date.toDateString() === now.toDateString();
  }

  function facts(id, rows) {
    $(id).replaceChildren(...rows.map(([label, value]) => {
      const row = create("div");
      row.append(create("dt", "", label), create("dd", "", value));
      return row;
    }));
  }

  function serviceEntries(snapshot) {
    return Object.entries(snapshot.live?.services || {}).filter(([, value]) => value?.enabled !== false);
  }

  function setConnection(mode, label) {
    $("observatory-connection").className = `connection ${mode}`;
    $("observatory-status").textContent = label;
  }

  function updateFreshness() {
    if (!state.snapshot) return;
    const captured = parsedDate(state.snapshot.captured_at);
    const age = captured ? Math.max(0, Date.now() - captured) : Infinity;
    const stale = age > STALE_MS || Date.now() - state.lastSuccessAt > STALE_MS || !$("observatory-error").hidden;
    const ageText = Number.isFinite(age) ? `${Math.floor(age / 1000)}s ago` : "at an unreported time";
    $("sample-freshness").textContent = `${stale ? "Stale sample" : "Sampled"} ${ageText}${state.live ? " · every 2s" : " · auto refresh paused"}`;
    setConnection(stale ? "stale" : "online", stale ? "Stale" : state.live ? "Live" : "Paused");
  }

  function updateUrl() {
    const params = new URLSearchParams(location.search);
    params.set("tab", state.tab);
    params.set("range", state.range);
    params.set("events", state.filter);
    history.replaceState(null, "", `${location.pathname}?${params}${location.hash}`);
  }

  function selectTab(tab, moveFocus = false) {
    state.tab = tab;
    for (const name of TABS) {
      const selected = tab === name;
      $(`tab-${name}`).setAttribute("aria-selected", String(selected));
      $(`tab-${name}`).tabIndex = selected ? 0 : -1;
      $(`view-${name}`).hidden = !selected;
    }
    updateTitle();
    if (moveFocus) $(`tab-${tab}`).focus({ preventScroll: true });
    updateUrl();
    requestAnimationFrame(drawWires);
  }

  function updateTitle() {
    document.title = `${$(`tab-${state.tab}`).textContent} — ${state.agent?.name || "Agent"} Observatory`;
  }

  function applyIdentity(identity) {
    if (!identity || typeof identity.name !== "string") return;
    state.agent = identity;
    $("agent-name").textContent = identity.name;
    $("agent-name").disabled = false;
    $("rename-button").disabled = false;
    $("identity-help").textContent = `Shared by this deployment. Takes effect on following turns. Account default: ${identity.default_name || "unreported"}.`;
    $("chat-title").textContent = `Chat with ${identity.name}`;
    $("chat-frame").title = `Chat with ${identity.name}`;
    updateTitle();
  }

  function openIdentity() {
    if (!state.agent || state.savingIdentity) return;
    $("identity-form").hidden = false;
    $("identity-error").hidden = true;
    $("agent-name-input").value = state.agent.name;
    $("agent-name-input").setAttribute("aria-invalid", "false");
    $("agent-name-input").focus();
    $("agent-name-input").select();
  }

  function closeIdentity() {
    if (state.savingIdentity) return;
    $("identity-form").hidden = true;
    $("agent-name").focus();
  }

  async function saveIdentity(event) {
    event.preventDefault();
    if (state.savingIdentity) return;
    const name = $("agent-name-input").value.trim();
    const show = message => {
      $("identity-error").textContent = message;
      $("identity-error").hidden = false;
    };
    if (!name || [...name].length > 64) {
      show("Use between 1 and 64 characters for the agent name.");
      $("agent-name-input").setAttribute("aria-invalid", "true");
      $("agent-name-input").focus();
      return;
    }
    state.savingIdentity = true;
    $("identity-error").hidden = true;
    $("identity-form").setAttribute("aria-busy", "true");
    for (const id of ["identity-save", "identity-cancel", "agent-name-input"]) $(id).disabled = true;
    const controller = new AbortController();
    const deadline = setTimeout(() => controller.abort(), 15_000);
    let saved = false;
    try {
      const response = await fetch("/api/agent-identity", {
        method: "PUT", headers: { Authorization: `Bearer ${state.token}`, "Content-Type": "application/json" },
        body: JSON.stringify({ name }), signal: controller.signal,
      });
      const identity = await response.json();
      if (!response.ok) {
        if (response.status === 400) $("agent-name-input").setAttribute("aria-invalid", "true");
        throw new Error(response.status === 401 || response.status === 403
          ? "Access expired. Return to chat and reopen the Observatory with the current link."
          : identity.error || "Agent name could not be saved. Try again.");
      }
      if (typeof identity.name !== "string") throw new Error("The name update could not be confirmed. Try saving again.");
      state.identityUpdatedAt = Date.now();
      applyIdentity(identity);
      $("observatory-announcer").textContent = `Agent name saved as ${identity.name}. Following turns use this identity.`;
      saved = true;
    } catch (error) {
      show(error.name === "AbortError" ? "The name update timed out. Try saving again to confirm it." : error.message || "The name update could not be confirmed. Try again.");
    } finally {
      clearTimeout(deadline);
      state.savingIdentity = false;
      $("identity-form").setAttribute("aria-busy", "false");
      for (const id of ["identity-save", "identity-cancel", "agent-name-input"]) $(id).disabled = false;
      if (saved) closeIdentity();
      else $("agent-name-input").focus();
    }
  }

  const SIGNALS = [["hearing", "Heard speech"], ["audio", "Sound input"], ["vision", "Image / video"], ["llm", "LLM"], ["speaking", "Speech output"], ["tools", "Tools"], ["documents", "Documents"]];
  function signalOf(item) {
    return item.signal || (item.category === "tools" ? "tools" : "llm");
  }
  function formatBytes(value) {
    const bytes = Number(value);
    const unit = bytes >= 1024 ** 3 ? 3 : bytes >= 1024 ** 2 ? 2 : bytes >= 1024 ? 1 : 0;
    return `${(bytes / 1024 ** unit).toFixed(unit ? 2 : 0)} ${["B", "KiB", "MiB", "GiB"][unit]}`;
  }
  function meter(label, value, maximum, detail) {
    const row = create("div", "instrument-bar");
    const track = create("div", "t");
    const fill = create("div", "f");
    track.append(fill);
    fill.style.width = `calc((100% - 4px) * ${Math.max(0, Math.min(1, value / Math.max(1, maximum)))})`;
    track.setAttribute("role", "meter");
    track.setAttribute("aria-label", label);
    track.setAttribute("aria-valuemin", "0");
    track.setAttribute("aria-valuemax", String(maximum));
    track.setAttribute("aria-valuenow", String(value));
    row.append(create("span", "bar-label", label), track, create("span", "bar-value", detail));
    return row;
  }
  function renderDisplay(snapshot) {
    renderWeightInstrument(snapshot);
    renderEventInstrument(snapshot);
    renderStateInstrument(snapshot);
    renderMemoryInstrument(snapshot);
    document.querySelectorAll(".visualization .d7").forEach(node => {
      const value = node.querySelector(".segment-value")?.textContent || node.textContent;
      const ghost = create("span", "segment-ghost", value.replace(/[0-9]/g, "8"));
      ghost.setAttribute("aria-hidden", "true");
      node.replaceChildren(ghost, create("span", "segment-value", value));
    });
    requestAnimationFrame(drawWires);
  }
  function renderWeightInstrument(snapshot) {
    const inventory = snapshot.weights || {};
    const components = inventory.components || [];
    $("loaded-graphs").textContent = numeric(inventory.loaded) ? String(inventory.loaded) : String(components.filter(item => item.state === "loaded").length);
    $("loaded-bytes").textContent = (Number(inventory.reported_weight_bytes || 0) / 1024 ** 3).toFixed(2);
    $("loaded-bytes").parentElement.hidden = !inventory.reported_weight_bytes;
    const reported = components.length ? components : serviceEntries(snapshot).map(([name, value]) => ({
      role: name, name: "Execution service", state: value.ok ? "reachable" : "unavailable", source: "HTTP health observation",
    }));
    $("component-network").replaceChildren(...reported.map(component => {
      const box = create("article", `readout component-node ${component.state === "loaded" ? "loaded" : ""}`);
      box.setAttribute("data-wire-node", "");
      box.append(create("span", "field-label", component.role), create("strong", "d14 lit component-state", component.state.toUpperCase()),
        create("p", "component-name", component.name), create("span", "muted", component.bytes ? formatBytes(component.bytes) : component.policy === "request" ? "Loads for each spoken reply" : component.source));
      if (typeof component.gpu_resident === "boolean") box.append(create("span", "muted", component.gpu_resident ? "GPU residency confirmed" : "GPU weights released"));
      return box;
    }));
    const daemon = snapshot.live?.daemon || {};
    const memory = snapshot.live?.environment?.memory || {};
    const bars = [];
    if (numeric(daemon.comprehension_context_tokens) && numeric(daemon.comprehension_context_ceiling)) bars.push(meter("Resident KV budget", Number(daemon.comprehension_context_tokens), Number(daemon.comprehension_context_ceiling), `${count(daemon.comprehension_context_tokens)} / ${count(daemon.comprehension_context_ceiling)} tokens`));
    if (numeric(memory.used_percent)) bars.push(meter("Host memory", Number(memory.used_percent), 100, `${Number(memory.used_percent).toFixed(1)}%${numeric(memory.used_gib) ? ` · ${memory.used_gib} GiB` : ""}`));
    $("weight-budget-bars").replaceChildren(...bars);
  }
  function observedActivity(snapshot) {
    const phases = new Map();
    const add = (phase, detail) => phases.set(phase, [...(phases.get(phase) || []), detail]);
    const harness = snapshot.live?.harness || {};
    if (harness.available && harness.fresh) {
      const descriptions = { listening: "Daemon waiting for input", hearing: "Daemon detecting input", thinking: "Daemon processing turn", speaking: "Daemon playing reply" };
      if (descriptions[harness.state]) add(harness.state, descriptions[harness.state]);
      if (harness.capture_fresh && !["muted", "paused", "offline"].includes(harness.state)) add("listening", "Daemon mic capturing");
    }
    const stagePhases = { queued: ["thinking", "Request queued"], comprehension: ["hearing", "Understanding media"], language: ["thinking", "Generating reply"], tts: ["speaking", "Synthesizing reply"] };
    for (const [stage, number] of Object.entries(snapshot.live?.activity?.stages || {})) {
      if (Number(number) > 0 && stagePhases[stage]) add(stagePhases[stage][0], `Portal · ${stagePhases[stage][1]}`);
    }
    const chat = Date.now() - state.chatObservedAt < 15000 ? state.chatActivity : null;
    if (chat) {
      for (const stage of new Set(chat.stages)) {
        if (stagePhases[stage]) add(stagePhases[stage][0], `Chat · ${stagePhases[stage][1]}`);
      }
      if (chat.microphone) add("listening", "Chat mic capturing");
      if (chat.hearing) add("hearing", "Chat detecting input");
      if (chat.playing) add("speaking", "Chat playing reply");
    }
    return { phases, chat, known: Boolean(chat || (harness.available && harness.fresh) || phases.size) };
  }
  function renderEventInstrument(snapshot) {
    const observed = observedActivity(snapshot);
    const current = ["speaking", "thinking", "hearing", "listening"].find(phase => observed.phases.has(phase));
    const activity = current ? current.toUpperCase() : observed.chat ? "CHAT READY" : "NO LIVE ACTIVITY";
    $("activity-label").textContent = activity.replaceAll(" ", "!");
    $("activity-label").setAttribute("aria-label", activity);
    $("activity-source").textContent = current ? observed.phases.get(current).join(" · ") : observed.chat ? "Text chat ready · chat microphone off" : "Waiting for an observed chat or voice activity";
    const now = Date.now();
    const from = now - state.window * 1000;
    const events = (snapshot.timeline || []).filter(item => parsedDate(item.at) && new Date(item.at).getTime() >= from && new Date(item.at).getTime() <= now);
    const counts = SIGNALS.slice(0, 6).map(([signal, label]) => [label, String(events.filter(item => signalOf(item) === signal).length)]);
    facts("signal-totals", counts);
    const focusId = document.activeElement?.dataset.eventId;
    const rows = SIGNALS.map(([signal, label]) => {
      const row = create("div", "event-lane");
      const track = create("div", "event-lane-track");
      const items = events.filter(item => signalOf(item) === signal);
      for (const item of items) {
        const marker = create("button", `event-mark ${item.status === "error" ? "error" : ""}`);
        marker.type = "button";
        marker.dataset.eventId = item.id;
        marker.style.left = `${Math.max(0, Math.min(98, (new Date(item.at).getTime() - from) / (state.window * 1000) * 98))}%`;
        marker.setAttribute("aria-label", `${item.label}, ${timestamp(item.at)}`);
        marker.addEventListener("click", () => {
          state.selectedEventId = item.id;
          $("raster-selection").textContent = `${timestamp(item.at)} · ${item.label}${item.detail ? ` · ${item.detail}` : ""} · ${item.source || "Source unreported"} · ${item.retention || "bounded diagnostic"}`;
        });
        track.append(marker);
      }
      row.append(create("span", "lane-label", label), track, create("span", "lane-count d7", String(items.length)));
      return row;
    });
    const axis = create("div", "raster-axis");
    axis.append(create("span", "", `${state.window < 60 ? state.window + "s" : state.window / 60 + "m"} ago`), create("span", "", "Now"));
    $("event-raster").replaceChildren(...rows, axis);
    if (focusId) $("event-raster").querySelector(`[data-event-id="${CSS.escape(focusId)}"]`)?.focus({ preventScroll: true });
    if (!events.some(item => item.id === state.selectedEventId)) {
      state.selectedEventId = null;
      $("raster-selection").textContent = `${events.length} observed events in this window. Select an event to inspect its source.`;
    }
    if (!events.length) $("raster-selection").textContent = "No events observed in this window. The timeline fills as inputs, inference stages, speech output, and tools arrive.";
  }
  function renderStateInstrument(snapshot) {
    const observed = observedActivity(snapshot);
    const phases = [["listening", "Listening"], ["hearing", "Hearing"], ["thinking", "Thinking"], ["speaking", "Speaking"]];
    $("state-flow").replaceChildren(...phases.map(([phase, label]) => {
      const details = observed.phases.get(phase);
      const idle = phase === "listening" && observed.chat && !observed.chat.microphone ? "Text chat ready · chat mic off" : observed.known ? "Idle" : "Unobserved";
      const box = create("div", `state-step frame ${details ? "is-current" : ""}`);
      box.append(create("span", "field-label", details ? "Current activity" : "Activity phase"), create("strong", "d14 lit", label.toUpperCase()), create("span", "muted", details ? details.join(" · ") : idle));
      return box;
    }));
    const queue = snapshot.live?.requests || {};
    const memory = snapshot.live?.environment?.memory || {};
    const meters = [];
    if (numeric(queue.active) && numeric(queue.slots)) meters.push(meter("Inference lanes", Number(queue.active), Number(queue.slots), `${queue.active} / ${queue.slots} active`));
    if (numeric(queue.inflight) && numeric(queue.limit)) meters.push(meter("Admission capacity", Number(queue.inflight), Number(queue.limit), `${queue.inflight} / ${queue.limit} admitted`));
    if (numeric(memory.used_percent)) meters.push(meter("Host RAM", Number(memory.used_percent), 100, `${Number(memory.used_percent).toFixed(1)}% used`));
    $("state-meters").replaceChildren(...meters);
    const transitions = (snapshot.timeline || []).filter(item => item.source === "voice-loop status" || ["request_received", "inference_stage", "audio_stream_started", "audio_stream_ended", "request_complete"].includes(item.event)).slice(0, 8).reverse();
    $("state-history").replaceChildren(...transitions.map(item => {
      const entry = create("div", "state-history-entry");
      entry.append(create("time", "", timestamp(item.at)), create("span", "", item.label));
      return entry;
    }));
    if (!transitions.length) $("state-history").append(create("p", "muted", "Chat and daemon transitions appear here as activity is observed."));
  }
  function renderMemoryInstrument(snapshot) {
    const memory = snapshot.memory || {};
    const passive = memory.passive || {};
    const stores = [["Passive memory", passive.available ? count(passive.entries) : "OFFLINE", "Durable · relevance-decayed"], ["Session memory", count(memory.session?.entries), "This browser only"], ["Working corpus", count(memory.virtual_context?.chunks), `${memory.virtual_context?.mode || "off"} · session scoped`]];
    $("store-readouts").replaceChildren(...stores.map(([label, value, detail]) => {
      const box = create("div", "frame store-readout");
      box.append(create("span", "field-label", label), create("strong", /^\d/.test(value) ? "d7 lit" : "store-status", value), create("span", "muted", detail));
      return box;
    }));
    const memories = (passive.recent || []).slice(0, 24).filter(item => parsedDate(item.created_at));
    const plot = $("memory-plot");
    const focusIndex = document.activeElement?.dataset.memoryIndex;
    plot.replaceChildren();
    if (memories.length) {
      const dates = memories.map(item => new Date(item.created_at).getTime());
      const min = Math.min(...dates), max = Math.max(...dates);
      $("memory-plot-range").textContent = `${timestamp(new Date(min).toISOString())} → ${timestamp(new Date(max).toISOString())}`;
      memories.forEach((item, index) => {
        const point = create("button", "memory-point");
        point.type = "button";
        point.dataset.memoryIndex = String(index);
        point.style.left = `${5 + (max === min ? .5 : (dates[index] - min) / (max - min)) * 90}%`;
        point.style.bottom = `${8 + Math.max(0, Math.min(1, Number(item.strength || 0))) * 82}%`;
        point.setAttribute("aria-label", `${item.kind || "Memory"}: ${item.text || "Retained source"}; strength ${item.strength}`);
        point.addEventListener("click", () => { $("memory-focus").textContent = `${item.text || "Retained memory"} · strength ${Number(item.strength || 0).toFixed(3)} · ${item.uses || 0} recalls · ${timestamp(item.created_at)}`; });
        plot.append(point);
      });
      plot.append(create("span", "plot-axis-label top", "Strength 1.0"), create("span", "plot-axis-label bottom", "0.0 · source time →"));
    } else {
      $("memory-plot-range").textContent = "No passive source points";
      plot.append(create("p", "plot-empty", passive.available ? "No retained passive memories yet. New memories appear here at their source time and recall strength." : "Passive store is offline. Session and working-corpus counts are shown above."));
    }
    if (focusIndex) plot.querySelector(`[data-memory-index="${focusIndex}"]`)?.focus({ preventScroll: true });
    const bars = [];
    if (numeric(passive.mean_strength) && passive.available) bars.push(meter("Mean recall strength", Number(passive.mean_strength), 1, Number(passive.mean_strength).toFixed(3)));
    if (numeric(memory.virtual_storage?.bytes)) {
      const storage = create("div", "storage-fact");
      storage.append(create("span", "", "Retained corpus storage"), create("span", "muted", `${formatBytes(memory.virtual_storage.bytes)} · ${count(memory.virtual_storage.corpora)} corpora`));
      bars.push(storage);
    }
    $("memory-budget-bars").replaceChildren(...bars);
  }

  function renderWeights(snapshot) {
    const daemon = snapshot.live?.daemon || {};
    $("model-name").textContent = daemon.model || document.body.dataset.model || "Unreported";
    $("model-source").textContent = daemon.available && daemon.model
      ? `Daemon report · ${daemon.state || "state unreported"} · ${timestamp(daemon.updated_at)}`
      : "Portal configuration · daemon model report unavailable";
    facts("weight-facts", [
      ["Runtime", daemon.state || "unavailable"],
      ["Resident context", numeric(daemon.comprehension_context_tokens) ? `${count(daemon.comprehension_context_tokens)} tokens` : "Unreported"],
      ["Context ceiling", numeric(daemon.comprehension_context_ceiling) ? `${count(daemon.comprehension_context_ceiling)} tokens` : "Unreported"],
      ["Loaded worker graphs", numeric(snapshot.weights?.loaded) ? String(snapshot.weights.loaded) : "Worker inventory unavailable"],
    ]);
    const entries = Object.entries(snapshot.live?.services || {});
    $("service-list").replaceChildren(...entries.map(([name, health]) => {
      const disabled = health?.enabled === false;
      const row = create("tr");
      const identity = create("th", "", name);
      identity.scope = "row";
      row.append(identity,
        create("td", !disabled && !health?.ok ? "error" : "", disabled ? "Disabled" : health?.ok ? "Reachable" : "Unavailable"),
        create("td", "", disabled ? "Not enabled in this deployment" : numeric(health?.status) ? `HTTP ${health.status} · health probe` : "No HTTP response"));
      return row;
    }));
    if (!entries.length) {
      const row = create("tr");
      const cell = create("td", "", "No service health observations in this snapshot.");
      cell.colSpan = 3;
      row.append(cell);
      $("service-list").append(row);
    }
  }

  function renderEvents(snapshot) {
    const filtered = (snapshot.timeline || []).filter(item =>
      ["conversation", "tools"].includes(item.category)
      && (state.filter === "all" || item.category === state.filter || signalOf(item) === state.filter) && inRange(item.at));
    const shown = filtered.slice(0, state.limit);
    $("timeline-list").replaceChildren(...shown.map(item => {
      const row = create("li", `timeline-item ${item.status === "error" ? "error" : ""}`);
      const time = create("time", "timeline-time", timestamp(item.at));
      if (parsedDate(item.at)) time.dateTime = item.at;
      const body = create("div", "timeline-body");
      body.append(create("strong", "", item.label || "Event"));
      if (item.detail) body.append(create("p", "", item.detail));
      const source = create("div", "timeline-source");
      source.append(create("span", "", SIGNALS.find(([signal]) => signal === signalOf(item))?.[1] || "Runtime event"),
        create("span", "", item.source || "Source unreported"), create("span", "", item.retention || "Retention unreported"));
      row.append(time, body, source);
      return row;
    }));
    $("timeline-empty").hidden = filtered.length > 0;
    $("timeline-more").hidden = shown.length >= filtered.length;
    $("timeline-more").textContent = `Show ${Math.min(PAGE_SIZE, filtered.length - shown.length)} more events`;
    $("timeline-meta").textContent = `${shown.length} of ${filtered.length} retained events`;
  }

  function renderStates(snapshot) {
    const live = snapshot.live || {};
    const harness = live.harness || {};
    const daemon = live.daemon || {};
    const environment = live.environment || {};
    const battery = environment.battery || {};
    const services = serviceEntries(snapshot);
    const memory = environment.memory?.used_percent;
    facts("live-readout", [
      ["Voice loop", harness.available ? `${harness.state || "unknown"}${harness.fresh ? "" : " · stale"}` : "Unavailable"],
      ["Voice source", timestamp(harness.updated_at)],
      ["Daemon microphone", harness.capture_fresh && !["muted", "paused", "offline"].includes(harness.state) ? "Capture frames arriving" : harness.state === "muted" ? "Muted" : harness.state === "paused" ? "Paused" : "Capture not recently observed"],
      ["Runtime", daemon.available ? daemon.state || "unknown" : "Unavailable"],
      ["Runtime source", timestamp(daemon.updated_at)],
      ["Services", services.length ? `${services.filter(([, value]) => value?.ok).length} / ${services.length} reachable` : "Unreported"],
      ["Inference", `${count(live.requests?.active)} active · ${count(live.requests?.queued)} queued`],
      ["Host RAM", numeric(memory) ? `${Number(memory).toFixed(1)}% used` : "Unreported"],
      ["Battery", battery.available ? `${battery.percentage}%${numeric(battery.voltage_v) ? ` · ${battery.voltage_v} V` : ""}` : "Unavailable"],
    ]);
    const work = snapshot.work || {};
    const tasks = work.live || [];
    $("work-current").replaceChildren(...tasks.map(task => {
      const card = create("article", "work-card");
      const header = create("header");
      const date = numeric(task.updated_at) ? new Date(Number(task.updated_at) * 1000).toISOString() : task.updated_at;
      header.append(create("span", "", String(task.status || "active").replaceAll("_", " ")), create("span", "", timestamp(date)));
      card.append(header, create("h3", "", task.objective || "Background objective"));
      if (task.current_stage) card.append(create("p", "", task.current_stage));
      const progress = create("div", "work-progress");
      progress.append(create("span", "", `Round ${count(task.round)}`), create("span", "", `${task.actions?.length || 0} actions`));
      card.append(progress);
      return card;
    }));
    $("work-empty").hidden = tasks.length > 0;
    const archive = work.archive || {};
    $("work-meta").textContent = `${archive.records_observed || 0} archived work records observed · bounded tail`;
    const archived = (archive.recent || []).slice(0, 12);
    $("archive-list").replaceChildren(...archived.map(item => {
      const row = create("li", "archive-item");
      row.append(create("span", "", item.status || "archived"), create("p", "", item.objective || "Archived objective"), create("time", "", timestamp(item.updated_at || item.created_at)));
      return row;
    }));
    if (!archived.length) $("archive-list").append(create("li", "panel-empty", archive.available ? "No records in the retained archive tail." : "Work archive unavailable."));
  }

  function renderMemory(snapshot) {
    const memory = snapshot.memory || {};
    const passive = memory.passive || {};
    const virtual = memory.virtual_context || {};
    facts("memory-stores", [
      ["Passive · durable host", passive.available ? `${count(passive.entries)} memories` : "Unavailable"],
      ["Session · this browser", `${count(memory.session?.entries)} entries`],
      ["Working context", `${count(virtual.chunks)} chunks · ${virtual.mode || "mode unreported"}`],
      ["Retained corpora · host", count(memory.virtual_storage?.corpora)],
    ]);
    const strength = passive.available && numeric(passive.mean_strength) ? Math.max(0, Math.min(1, Number(passive.mean_strength))) : null;
    $("memory-strength-fill").style.width = `calc((100% - 4px) * ${strength ?? 0})`;
    $("memory-strength-value").textContent = strength === null ? "—" : strength.toFixed(3);
    $("memory-meta").textContent = passive.available ? `${count(passive.entries)} durable memories · latest ${Math.min(16, passive.recent?.length || 0)} shown` : passive.reason || "Passive memory unavailable";
    const memories = (passive.recent || []).slice(0, 16);
    $("memory-list").replaceChildren(...memories.map(item => {
      const row = create("li", "memory-item");
      row.append(create("p", "", item.text || "Retained memory"));
      const vitals = create("div", "memory-vitals");
      vitals.append(create("span", "", item.kind || "turn"), create("time", "", timestamp(item.created_at)),
        create("span", "", numeric(item.strength) ? `Strength ${Number(item.strength).toFixed(3)}` : "Strength unreported"), create("span", "", `${count(item.uses)} recalls`), create("span", "", "Passive semantic memory · durable host"));
      row.append(vitals);
      return row;
    }));
    $("memory-empty").hidden = memories.length > 0;
    $("memory-empty").textContent = passive.available ? "No retained passive memories in this snapshot." : "Passive memory is unavailable. Session and working-context counts remain separate above.";
  }

  function showError(message) {
    $("observatory-error-message").textContent = message;
    $("observatory-error").hidden = false;
    $("review-loading").hidden = true;
    $("review").setAttribute("aria-busy", "false");
    setConnection(state.snapshot ? "stale" : "offline", state.snapshot ? "Stale" : "Offline");
    if (state.snapshot) updateFreshness();
    else $("sample-freshness").textContent = "No runtime sample received";
  }

  async function refresh({ announce = false } = {}) {
    if (state.refreshing) return;
    if (!state.token) {
      showError("Access is missing. Return to chat and open the Observatory from your authenticated session.");
      return;
    }
    state.refreshing = true;
    $("refresh-button").disabled = true;
    $("retry-button").disabled = true;
    const controller = new AbortController();
    state.controller = controller;
    const deadline = setTimeout(() => controller.abort(), 25_000);
    const startedAt = Date.now();
    try {
      const response = await fetch("/api/observatory", { headers: { Authorization: `Bearer ${state.token}` }, signal: controller.signal, cache: "no-store" });
      if (response.status === 401 || response.status === 403) throw new Error("Access expired or was denied. Return to chat and reopen the Observatory with the current access link.");
      if (!response.ok) throw new Error(`Runtime returned HTTP ${response.status}. Try refreshing again.`);
      const snapshot = await response.json();
      if (snapshot.schema !== "robit.omni.observatory.v1") throw new Error("The runtime returned an unsupported snapshot. Reload after updating the portal.");
      state.snapshot = snapshot;
      state.lastSuccessAt = Date.now();
      if (!state.savingIdentity && startedAt >= state.identityUpdatedAt) applyIdentity(snapshot.agent);
      $("observatory-error").hidden = true;
      renderDisplay(snapshot);
      renderWeights(snapshot);
      renderEvents(snapshot);
      renderStates(snapshot);
      renderMemory(snapshot);
      $("review-loading").hidden = true;
      $("review").setAttribute("aria-busy", "false");
      updateFreshness();
      if (announce) $("observatory-announcer").textContent = `Runtime sample refreshed. ${timestamp(snapshot.captured_at)}.`;
    } catch (error) {
      showError(error.name === "AbortError" ? "The runtime sample timed out. Try again when the service is reachable." : error.message || "The runtime could not be reached. Try again.");
    } finally {
      clearTimeout(deadline);
      if (state.controller === controller) state.controller = null;
      state.refreshing = false;
      $("refresh-button").disabled = false;
      $("retry-button").disabled = false;
    }
  }

  function drawWires() {
    document.querySelectorAll(".connected-stage").forEach(stage => {
      const svg = stage.querySelector(".wires");
      if (stage.offsetParent === null || !svg) return;
      const root = stage.getBoundingClientRect();
      const hub = stage.querySelector("[data-wire-hub]")?.getBoundingClientRect();
      if (!hub) return;
      svg.setAttribute("viewBox", `0 0 ${root.width} ${root.height}`);
      svg.replaceChildren();
      stage.querySelectorAll("[data-wire-node]").forEach(node => {
        const rect = node.getBoundingClientRect();
        const x = rect.left - root.left + rect.width / 2;
        const y = rect.top - root.top;
        const hx = hub.left - root.left + hub.width / 2;
        const hy = hub.bottom - root.top;
        const middle = hy + Math.max(8, (y - hy) / 2);
        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        path.setAttribute("d", `M${hx},${hy}V${middle}H${x}V${y}`);
        svg.append(path);
      });
    });
  }

  function initialize() {
    const supplied = new URLSearchParams(location.hash.slice(1)).get("access");
    state.token = supplied || "";
    try {
      if (supplied) sessionStorage.setItem("omni_access", supplied);
      else state.token = sessionStorage.getItem("omni_access") || "";
    } catch (_error) { /* A supplied capability still works when storage is blocked. */ }
    if (supplied) $("chat-link").href = `/#${new URLSearchParams({ access: supplied })}`;
    const chatUrl = new URL("/", location.href);
    chatUrl.searchParams.set("surface", "observatory");
    if (state.token) chatUrl.hash = new URLSearchParams({ access: state.token }).toString();
    $("chat-link").href = chatUrl.href;
    $("chat-link").target = "_blank";
    $("chat-link").rel = "noopener noreferrer";
    $("chat-new-tab").href = chatUrl.href;
    document.querySelectorAll(".open-chat").forEach(button => button.addEventListener("click", () => {
      $("observatory-chat").hidden = false;
      $("observatory-workspace").classList.add("chat-open");
      document.querySelectorAll(".open-chat").forEach(item => item.setAttribute("aria-expanded", "true"));
      if (!$("chat-frame").getAttribute("src")) $("chat-frame").src = chatUrl.href;
      $("close-chat").focus();
      requestAnimationFrame(drawWires);
    }));
    $("close-chat").addEventListener("click", () => {
      state.chatActivity = null;
      state.chatObservedAt = 0;
      $("chat-frame").removeAttribute("src");
      $("observatory-chat").hidden = true;
      $("observatory-workspace").classList.remove("chat-open");
      document.querySelectorAll(".open-chat").forEach(item => item.setAttribute("aria-expanded", "false"));
      document.querySelector(".open-chat").focus();
      if (state.snapshot) {
        renderStateInstrument(state.snapshot);
        renderEventInstrument(state.snapshot);
      }
      requestAnimationFrame(drawWires);
    });
    $("chat-frame").addEventListener("load", () => {
      $("chat-frame").contentWindow?.postMessage({ type: "omni-observatory-observe" }, location.origin);
    });
    window.addEventListener("message", event => {
      if (event.origin !== location.origin || event.source !== $("chat-frame").contentWindow
          || $("observatory-chat").hidden || event.data?.type !== "omni-observatory-activity") return;
      const data = event.data;
      state.chatActivity = { stages: Array.isArray(data.stages) ? data.stages.slice(0, 16).filter(stage => ["queued", "comprehension", "language", "tts"].includes(stage)) : [],
        microphone: data.microphone === true, hearing: data.hearing === true, playing: data.playing === true };
      state.chatObservedAt = Date.now();
      if (state.snapshot) {
        renderStateInstrument(state.snapshot);
        renderEventInstrument(state.snapshot);
      }
    });
    $("copy-chat-link").addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(chatUrl.href);
        $("chat-link-status").textContent = "Chat link copied.";
      } catch (_error) {
        $("chat-link-status").textContent = "Clipboard unavailable. Use Open in new tab to open the authenticated chat.";
      }
    });
    selectTab(state.tab);
    $("agent-name").addEventListener("dblclick", openIdentity);
    $("agent-name").addEventListener("keydown", event => {
      if (!["Enter", " ", "F2"].includes(event.key)) return;
      event.preventDefault();
      openIdentity();
    });
    $("rename-button").addEventListener("click", openIdentity);
    $("identity-form").addEventListener("submit", saveIdentity);
    $("identity-cancel").addEventListener("click", closeIdentity);
    $("identity-form").addEventListener("keydown", event => {
      if (event.key === "Escape") { event.preventDefault(); closeIdentity(); }
    });
    $("agent-name-input").addEventListener("input", () => {
      $("agent-name-input").setAttribute("aria-invalid", "false");
      $("identity-error").hidden = true;
    });
    TABS.forEach((tab, index) => {
      $(`tab-${tab}`).addEventListener("click", () => selectTab(tab));
      $(`tab-${tab}`).addEventListener("keydown", event => {
        const targets = { ArrowRight: (index + 1) % TABS.length, ArrowLeft: (index + TABS.length - 1) % TABS.length, Home: 0, End: TABS.length - 1 };
        if (!(event.key in targets)) return;
        event.preventDefault();
        selectTab(TABS[targets[event.key]], true);
      });
    });
    for (const kind of ["range", "filter"]) {
      document.querySelectorAll(`[data-${kind}]`).forEach(button => {
        button.setAttribute("aria-pressed", String(button.dataset[kind] === state[kind]));
        button.addEventListener("click", () => {
          state[kind] = button.dataset[kind];
          state.limit = PAGE_SIZE;
          document.querySelectorAll(`[data-${kind}]`).forEach(item => item.setAttribute("aria-pressed", String(item === button)));
          updateUrl();
          if (state.snapshot) renderEvents(state.snapshot);
        });
      });
    }
    $("timeline-more").addEventListener("click", () => {
      state.limit += PAGE_SIZE;
      if (state.snapshot) renderEvents(state.snapshot);
    });
    $("refresh-button").addEventListener("click", () => refresh({ announce: true }));
    $("retry-button").addEventListener("click", () => refresh({ announce: true }));
    $("live-toggle").addEventListener("click", () => {
      state.live = !state.live;
      $("live-toggle").setAttribute("aria-pressed", String(state.live));
      $("live-toggle").textContent = state.live ? "Auto refresh" : "Refresh paused";
      updateFreshness();
      if (state.live) refresh();
    });

    document.querySelectorAll("[data-window]").forEach(button => button.addEventListener("click", () => {
      state.window = Number(button.dataset.window);
      document.querySelectorAll("[data-window]").forEach(item => item.setAttribute("aria-pressed", String(item === button)));
      if (state.snapshot) renderEventInstrument(state.snapshot);
    }));
    new ResizeObserver(drawWires).observe($("review"));
    document.fonts.ready.then(drawWires);
    document.addEventListener("visibilitychange", () => { if (!document.hidden && state.live) refresh(); });
    setInterval(() => { if (state.live && !document.hidden) refresh(); }, REFRESH_MS);
    setInterval(updateFreshness, 1000);
    refresh();
  }

  initialize();
})();
