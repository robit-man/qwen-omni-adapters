/* Typed, source-backed memory structure. Geometry carries no similarity score. */
(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const make = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const number = value => Number.isFinite(Number(value)) ? Number(value).toLocaleString() : "—";
  const when = value => {
    const date = typeof value === "number" ? new Date(value * 1000) : new Date(value);
    return value && !Number.isNaN(date.getTime()) ? date.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "Not recorded";
  };
  let scope = new URLSearchParams(location.search).get("memory") === "session" ? "session" : "durable";
  let selected = null, model = null, snapshot = null;
  const node = (id, label, value, unit, column, row, facts, records = []) => ({ id, label, value, unit, column, row, facts, records });

  function durableModel(memory) {
    const source = memory.durable || {}, cache = memory.passive || {};
    const known = source.available === true;
    const nodes = [node("sources", "Source archive", known ? source.events : null, "committed events", 0, 1,
      [["Scope", "Device archive · aggregate view"], ["First source", when(source.oldest_at)], ["Latest source", when(source.newest_at)],
        ["With semantic vectors", number(source.indexed_events)], ["Without semantic vectors", number(source.unindexed_events)]])];
    const labels = { exchange: "Exchanges", heard: "Heard fragments", sound: "Sound observations", turn: "Legacy turns" };
    const kinds = Object.entries(source.kinds || {}).slice(0, 4);
    const links = [];
    kinds.forEach(([kind, count], index) => {
      const id = `kind-${kind}`;
      nodes.push(node(id, labels[kind] || kind, count, "source events", 1, index,
        [["Record kind", kind], ["Source", "Committed archive events"], ["Relationship", "Membership in the source archive"],
          ["Attribution", kind === "heard" ? "Retained speech; identity and addressee remain as recorded" : "Original source attribution is preserved"]]));
      links.push({ from: "sources", to: id, kind: "contains", count });
    });
    const windows = source.recent_windows || [];
    nodes.push(node("temporal", "Temporal views", known ? source.windows : null, "hour / day windows", 2, 0,
      [["Ready windows", known ? number(source.windows - source.pending_windows) : "Unavailable"], ["Awaiting refresh", number(source.pending_windows)],
        ["Relationship", "Extractive views retain source IDs and hashes"], ["Coverage", `Latest ${windows.length} windows shown`]],
      windows.map(item => ({ title: `${item.unit} · ${when(item.start_at)}`, text: `${item.pending ? "Refresh pending" : "Current view"} · ${item.sources === null ? "source membership not built" : `${number(item.sources)} cached source references`}` }))));
    if (known && source.windows > source.pending_windows) links.push({ from: "sources", to: "temporal", kind: "source-linked views", count: source.windows - source.pending_windows });
    const references = new Map((source.cluster_sources || []).map(item => [item.memory_id, item.sources]));
    nodes.push(node("semantic", "Semantic cache", cache.available ? cache.entries : null, "derived memories", 2, Math.max(2, kinds.length - 1),
      [["Source", "Derived semantic cache"], ["Cache strength", cache.entries ? Number(cache.mean_strength || 0).toFixed(3) : "No cache entries"],
        ["Recalls", number(cache.recalls)], ["Indexing", known ? `${number(source.unindexed_events)} source events without semantic vectors` : "Source index unavailable"],
        ["Strength means", "Cache retention weight; not factual confidence"]],
      (cache.recent || []).slice(0, 16).map(item => ({ title: `${item.kind} · ${when(item.created_at)}`, text: item.text,
        meta: `${number(item.uses)} recalls · ${references.has(item.id) ? `${number(references.get(item.id))} linked sources` : "Source linkage not reported"}` }))));
    const linked = [...references.values()].reduce((sum, count) => sum + count, 0);
    if (linked) links.push({ from: "sources", to: "semantic", kind: "derived from", count: linked });
    return {
      nodes, links, title: "Durable memory structure",
      note: known ? `${number(source.events)} sources retained · ${number(source.unindexed_events)} without semantic vectors` : source.reason || "Source archive unavailable",
      metrics: [["Source events", known ? source.events : null, "Committed before indexing"], ["Semantic cache", cache.available ? cache.entries : null, "Derived · relevance-decayed"], ["Temporal views", known ? source.windows : null, `${number(source.pending_windows)} awaiting refresh`]],
      empty: known && source.events === 0 ? "The source archive is available and has no committed events." : null,
    };
  }

  function sessionModel(memory) {
    const conversation = memory.conversation || {}, notes = memory.session || {}, context = memory.virtual_context || {};
    const turns = conversation.recent || [];
    const nodes = [
      node("browser", "This browser", conversation.messages ?? 0, "retained messages", 0, 1,
        [["Scope", "This browser session only"], ["Retained turns", number(conversation.turns)], ["Running turns", number(conversation.active)],
          ["Conversation limit", `${number(conversation.limit)} retained turns`], ["Saved notes", number(notes.entries)]]),
      node("conversation", "Conversation", conversation.turns ?? 0, "retained turns", 1, 0,
        [["Source", "Browser continuation journal"], ["Messages", number(conversation.messages)], ["Running turns", number(conversation.active)], ["Coverage", `Latest ${turns.length} retained turns shown`]],
        [...turns].reverse().map(item => ({ title: `Turn ${item.number} · ${when(item.started_at)} · ${item.status}`,
          text: item.user ? `You: ${item.user}` : "No text input retained", reply: item.assistant ? `Agent: ${item.assistant}` : "Reply not retained yet" }))),
      node("notes", "Saved notes", notes.entries ?? 0, "session entries", 1, 1,
        [["Source", "Explicit session memory writes"], ["Characters", number(notes.chars)], ["Lifetime", `${number(notes.ttl_seconds)} seconds idle`],
          ["Scope", "This browser; cleared with Trash"]],
        (notes.recent || []).map(item => ({ title: `${item.topic} / ${item.key}`, text: item.value, meta: when(item.saved_at) }))),
      node("context", "Working context", context.chunks ?? 0, "indexed chunks", 1, 2,
        [["Mode", context.mode || "Unavailable"], ["Documents", number(context.documents)], ["Embeddings", number(context.embeddings)],
          ["Physical context", `${number(context.physical_context_tokens)} tokens`], ["Scope", "This browser's working corpus"]]),
      node("entities", "Entity index", context.entities ?? 0, "stored entities", 2, 1,
        [["Source", "Working-corpus entity index"], ["Entities", number(context.entities)], ["Meaning", "Entities reported by this session's corpus; individual edges are not exposed here"]]),
      node("relationships", "Relation index", context.relationships ?? 0, "stored relations", 2, 2,
        [["Source", "Working-corpus relation index"], ["Relations", number(context.relationships)], ["Meaning", "Stored relation count; no inferred causal links are drawn"]]),
    ];
    const links = ["conversation", "notes", "context"].map(to => ({ from: "browser", to, kind: "session store", count: null }));
    if (context.entities > 0) links.push({ from: "context", to: "entities", kind: "entity index", count: context.entities });
    if (context.relationships > 0) links.push({ from: "context", to: "relationships", kind: "relation index", count: context.relationships });
    return { nodes, links, title: "Session memory structure", note: "Conversation, saved notes, and working context keep their own lifetimes.",
      metrics: [["Conversation", conversation.turns ?? 0, "Retained turns · this browser"], ["Saved notes", notes.entries ?? 0, "Explicit session memory"], ["Working context", context.chunks ?? 0, `${number(context.entities)} entities · ${number(context.relationships)} relations`]],
      empty: !conversation.turns && !notes.entries && !context.chunks ? "This browser has no retained turns, saved notes, or working chunks." : null };
  }

  function episodicModel(memory, selectedScope) {
    const service = memory.episodic || {}, data = service[selectedScope === "durable" ? "device" : "session"] || {};
    const unavailable = data.available === false;
    const visible = data.content_available === true, sources = data.recent || [], facts = data.recent_facts || [];
    const records = sources.map(item => ({ title: `${item.modality} · ${when(item.occurred_at)}`,
      text: item.text, meta: `${item.authority} · ${item.actor} · source ${item.event_id}${item.text_truncated ? " · excerpt" : ""}` }));
    const root = node("experience", "Source events", unavailable ? null : data.events ?? 0, "immutable records", 0, 1,
      [["Scope", selectedScope === "durable" ? "Device owner archive" : "This browser only"], ["Episodes", number(data.episodes)],
        ["Semantic vectors", number(data.vectors)], ["Awaiting graph encoding", number(data.encoding_pending)],
        ["Awaiting vectors", number(data.vector_pending)], ["Attribution", "Recorded authority and speaker identity remain separate"],
        ["Contents", visible ? "Latest source excerpts; no raw media or reasoning" : "Owner content is private; aggregate structure only"]], records);
    const nodes = [root], links = [];
    const relationships = facts.filter(item => !item.valid_to);
    const names = [...new Set(relationships.flatMap(item => [item.subject, item.object]))].slice(0, 6);
    const identity = new Map(names.map(name => [name, `entity-${encodeURIComponent(name)}`]));
    names.forEach((name, index) => {
      const connected = facts.filter(item => item.subject === name || item.object === name);
      const supports = new Set(connected.flatMap(item => item.support.map(source => source.event_id)));
      const id = identity.get(name);
      nodes.push(node(id, name, connected.length, "source-backed relations", index % 2 + 1, Math.floor(index / 2),
        [["Entity", name], ["Supporting sources", number(supports.size)], ["Meaning", "Relations are model-derived; inspect the quoted evidence"], ["View", "Recent graph neighborhood; layout carries no similarity score"]],
        connected.map(fact => ({ title: `${fact.subject} · ${fact.predicate} · ${fact.object}`,
          text: fact.support.map(source => `Source ${source.event_id}: “${source.quote}”`).join("\n"),
          meta: `${fact.relation_type} · ${fact.epistemic_status} · ${fact.authority} · ${when(fact.valid_from)}${fact.valid_to ? ` → ${when(fact.valid_to)} · superseded` : " · active"}` }))));
      links.push({ from: "experience", to: id, kind: "quoted source support", count: supports.size });
    });
    const drawn = new Set();
    relationships.forEach(fact => {
      const from = identity.get(fact.subject), to = identity.get(fact.object);
      const key = `${from}:${to}:${fact.predicate}`;
      if (from && to && from !== to && !drawn.has(key)) {
        links.push({ from, to, kind: `${fact.relation_type} · ${fact.predicate}`, count: null }); drawn.add(key);
      }
    });
    if (!names.length) {
      Object.entries(data.modalities || {}).slice(0, 6).forEach(([modality, count], index) => {
        const id = `modality-${modality}`;
        nodes.push(node(id, modality, count, "source events", index % 2 + 1, Math.floor(index / 2),
          [["Modality", modality], ["Graph encoding", "Runs in the background and yields to active conversation"], ["Source retention", "Original events are available before graph encoding"]],
          records.filter((_, i) => sources[i].modality === modality)));
        links.push({ from: "experience", to: id, kind: "source modality", count });
      });
    }
    if (nodes.length === 1) nodes.push(node("encoding", "Memory encoder", data.pending ?? 0, "pending sources", 1, 1,
      [["State", service.encoding_active ? "Encoding" : "Waiting for source activity"], ["Recall", "Original source search works before encoding"], ["Scope", selectedScope === "durable" ? "Device-owner archive; preserved through browser Trash" : "This browser; cleared with Trash"]]));
    const pending = Number(data.pending || 0);
    const status = unavailable ? "Memory store unavailable" : service.last_encoding_error ? `Encoding deferred · ${service.last_encoding_error.replaceAll("_", " ")}`
      : service.encoding_active ? "Encoding in background" : pending ? "Pending · foreground activity takes priority" : "Index current";
    return { nodes, links, title: selectedScope === "durable" ? "Durable experience graph" : "Session experience graph",
      note: `${status} · ${number(pending)} pending`,
      metrics: [["Source events", unavailable ? null : data.events ?? 0, `${number(data.episodes)} episodes · preserved originals`],
        ["Graph relations", unavailable ? null : data.active_facts ?? 0, `${number(data.entities)} entities · ${number(data.facts)} including history`],
        ["Semantic index", unavailable ? null : data.vectors ?? 0, `${number(data.vector_pending)} awaiting vectors`]],
      empty: unavailable ? "This memory store is temporarily unavailable. Retrying with the next live update." : !data.events ? "No source events in this scope yet. New interactions are committed before background encoding." : null };
  }

  function details() {
    const item = model.nodes.find(item => item.id === selected) || model.nodes[0];
    selected = item.id;
    $("memory-detail-title").textContent = item.label;
    const facts = item.facts.map(([key, value]) => {
      const row = make("div"); row.append(make("dt", "", key), make("dd", "", value)); return row;
    });
    $("memory-node-facts").replaceChildren(...facts);
    $("memory-node-records").replaceChildren(...item.records.map(record => {
      const entry = make("li", "memory-record");
      entry.append(make("strong", "", record.title), make("p", "", record.text));
      if (record.reply) entry.append(make("p", "", record.reply));
      if (record.meta) entry.append(make("span", "muted", record.meta));
      return entry;
    }));
    $("memory-record-empty").hidden = item.records.length > 0;
    $("memory-record-empty").textContent = item.id === "semantic" ? snapshot.memory?.passive?.entries > 0 && snapshot.memory.passive.content_available === false ? "Cache contents stay in the owner voice session. This view shows aggregate structure." : "No derived cache entries. Source events remain in the archive."
      : item.id === "notes" ? "No saved notes in this browser session. Conversation turns are listed separately."
      : item.id === "conversation" ? "No retained conversation turns in this browser session."
      : scope === "durable" ? "Device archive contents remain in the owner voice session. This view shows aggregate structure."
      : "No source excerpts for this node yet. New events appear here as they are committed.";
    document.querySelectorAll("[data-memory-node]").forEach(button => {
      button.setAttribute("aria-pressed", String(button.dataset.memoryNode === selected));
    });
    $("memory-edge-list").replaceChildren(...model.links.filter(link => link.from === selected || link.to === selected).map(link => {
      const from = model.nodes.find(item => item.id === link.from), to = model.nodes.find(item => item.id === link.to);
      return make("li", "", `${from.label} → ${to.label} · ${link.kind}${link.count === null ? "" : ` · ${number(link.count)}`}`);
    }));
    if (!$("memory-edge-list").childElementCount) $("memory-edge-list").append(make("li", "muted", "No source links reported for this node."));
  }

  function draw() {
    const graph = $("memory-graph");
    if (!graph || !model || graph.offsetParent === null) return;
    const narrow = graph.clientWidth < 600;
    graph.style.gridTemplateColumns = `repeat(${narrow ? 2 : 3}, minmax(0, 1fr))`;
    for (const [index, item] of model.nodes.entries()) {
      const button = [...graph.querySelectorAll("[data-memory-node]")].find(button => button.dataset.memoryNode === item.id);
      button.style.gridColumn = narrow ? index === 0 ? "1 / -1" : String((index - 1) % 2 + 1) : String(item.column + 1);
      button.style.gridRow = !narrow && index === 0 ? `1 / span ${Math.max(2, Math.ceil((model.nodes.length - 1) / 2))}` : String(narrow ? index === 0 ? 1 : Math.floor((index - 1) / 2) + 2 : item.row + 1);
    }
    const svg = $("memory-links"), root = graph.getBoundingClientRect();
    svg.setAttribute("viewBox", `0 0 ${root.width} ${root.height}`);
    svg.replaceChildren(...model.links.map(link => {
      const buttons = [...graph.querySelectorAll("[data-memory-node]")];
      const a = buttons.find(button => button.dataset.memoryNode === link.from).getBoundingClientRect();
      const b = buttons.find(button => button.dataset.memoryNode === link.to).getBoundingClientRect();
      const horizontal = a.right + 4 < b.left;
      const x1 = (horizontal ? a.right : a.left + a.width / 2) - root.left;
      const y1 = (horizontal ? a.top + a.height / 2 : a.bottom) - root.top;
      const x2 = (horizontal ? b.left : b.left + b.width / 2) - root.left;
      const y2 = (horizontal ? b.top + b.height / 2 : b.top) - root.top;
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", horizontal ? `M${x1},${y1}H${(x1 + x2) / 2}V${y2}H${x2}` : `M${x1},${y1}V${(y1 + y2) / 2}H${x2}V${y2}`);
      path.classList.toggle("is-selected", link.from === selected || link.to === selected);
      return path;
    }));
  }

  function render(value) {
    if (!$("memory-graph")) return;
    snapshot = value;
    model = value.memory?.episodic?.enabled ? episodicModel(value.memory || {}, scope) : scope === "durable" ? durableModel(value.memory || {}) : sessionModel(value.memory || {});
    if (!model.nodes.some(item => item.id === selected)) selected = model.nodes[0].id;
    const focus = document.activeElement?.dataset.memoryNode;
    $("memory-structure-title").textContent = model.title;
    $("memory-index-status").textContent = model.note;
    $("memory-meta").textContent = scope === "durable" ? "Device archive · aggregate source structure" : "This browser only · retained context";
    $("memory-structure-empty").hidden = !model.empty;
    $("memory-structure-empty").textContent = model.empty || "";
    $("store-readouts").replaceChildren(...model.metrics.map(([label, count, description]) => {
      const item = make("div", "store-readout");
      item.append(make("span", "field-label", label), make("strong", count === null ? "store-status" : "d7 lit", count === null ? "OFFLINE" : number(count)), make("span", "muted", description));
      return item;
    }));
    const buttons = model.nodes.map(item => {
      const button = make("button", "memory-graph-node frame"); button.type = "button"; button.dataset.memoryNode = item.id;
      button.append(make("span", "memory-node-name", item.label), make("strong", item.value === null ? "memory-node-unavailable" : "d7 lit", item.value === null ? "—" : number(item.value)), make("span", "muted", item.unit));
      button.addEventListener("click", () => { selected = item.id; details(); draw(); $("memory-selection-status").textContent = `Selected ${item.label}. Details updated.`; });
      return button;
    });
    $("memory-graph").querySelectorAll("[data-memory-node]").forEach(item => item.remove());
    $("memory-graph").append(...buttons);
    details();
    if (focus) buttons.find(button => button.dataset.memoryNode === focus)?.focus({ preventScroll: true });
    document.querySelectorAll("[data-memory-scope]").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.memoryScope === scope)));
    requestAnimationFrame(draw);
  }

  document.querySelectorAll("[data-memory-scope]").forEach(button => button.addEventListener("click", () => {
    scope = button.dataset.memoryScope; selected = null;
    const params = new URLSearchParams(location.search); params.set("memory", scope);
    history.replaceState(null, "", `${location.pathname}?${params}${location.hash}`);
    if (snapshot) render(snapshot);
  }));
  if ($("memory-graph")) new ResizeObserver(() => requestAnimationFrame(draw)).observe($("memory-graph"));
  document.fonts?.ready.then(draw);
  window.OmniMemoryGraph = { render, draw };
})();
