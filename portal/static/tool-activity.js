/* Request-linked dispatch receipts. No timing-based or inferred task chains. */
(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const make = (tag, cls, text) => { const node = document.createElement(tag); node.className = cls; node.textContent = text; return node; };
  let events = [], requestId = "", selected = null, optionsKey = "";
  let token = "", drawKey = "";
  let generation = 0;
  const expanded = new Set(), cache = new Map(), pending = new Map();
  const time = value => value ? new Date(value).toLocaleTimeString() : "Time not retained";
  async function loadDetails(item, target) {
    const startedGeneration = generation;
    const key = `${item.request_id}:${item.tool_round}:${item.tool_call_id}`;
    if (!item.request_id || !item.tool_call_id || item.tool_round == null) {
      target.textContent = "This older metadata has no retained call contents. New calls include inspectable inputs and outputs."; return;
    }
    try {
      let entry = cache.get(key);
      if (!entry || Date.now() - entry.loaded > 5000 || (entry.data.status === "running" && item.event === "tool_call_completed")) {
        if (!pending.has(key)) {
          const query = new URLSearchParams({request_id:item.request_id, round:String(item.tool_round), call_id:item.tool_call_id});
          pending.set(key, (async () => {
            const controller = new AbortController(), timer = setTimeout(() => controller.abort(), 10000);
            try {
              const response = await fetch(`/api/observatory/tool-call?${query}`, {headers:{Authorization:`Bearer ${token}`}, cache:"no-store", signal:controller.signal});
              if (!response.ok) throw new Error(response.status === 404 ? "Call details belong to the calling session or have expired." : "Call details are temporarily unavailable.");
              const value = {data:await response.json(), loaded:Date.now()};
              if (startedGeneration !== generation) throw new Error("Call details cleared.");
              cache.set(key,value);
              while (cache.size > 64) cache.delete(cache.keys().next().value);
              return value;
            } finally { clearTimeout(timer); }
          })());
        }
        try { entry = await pending.get(key); } finally { pending.delete(key); }
      }
      if (!target.isConnected || startedGeneration !== generation) return;
      const data = entry.data, status = make("p", "scope-note", data.status === "running" ? "Call running; output will update when it returns." : data.ok ? "Call returned successfully. Verify the task's acceptance criteria separately." : "Call returned a failure.");
      const args = make("pre", "tool-call-content", data.arguments || "{}"), output = make("pre", "tool-call-content", data.output || "No output returned yet.");
      for (const [node,label] of [[args,"Call arguments"],[output,"Returned tool output"]]) {node.tabIndex=0;node.setAttribute("role","region");node.setAttribute("aria-label",label);}
      target.replaceChildren(status, make("h4", "", "Arguments"), args,
        make("h4", "", "Returned output"), output,
        make("p", "scope-note", `${data.arguments_truncated || data.output_truncated ? "Large content is truncated. " : ""}${data.retention}`));
    } catch (error) { cache.delete(key); if (target.isConnected) target.textContent = error.name === "AbortError" ? "Call inspection timed out. Close and reopen to retry." : error.message; }
  }
  function inspect(item) {
    selected = item.id;
    $("tool-receipt-title").textContent = item.detail || "Tool dispatch";
    const fields = [["Signal", item.event === "tool_call_started" ? "Dispatch started" : item.status === "error" ? "Failed return" : "Returned"],
      ["Source time", item.at || "Not retained"], ["Request", item.request_id || "Not retained; no chain asserted"],
      ["Tool round", item.tool_round ?? "Not retained"], ["Source", item.source], ["Retention", item.retention]];
    $("tool-receipt-facts").replaceChildren(...fields.map(([label, value]) => { const row = document.createElement("div"); row.append(make("dt", "", label), make("dd", "", String(value || "Not retained"))); return row; }));
    $("tool-chain").querySelectorAll("button").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.receipt === selected)));
  }
  function draw() {
    const key = JSON.stringify([requestId,events.map(item=>[item.id,item.status,item.tool_call_id])]);
    if (drawKey === key) return;
    drawKey = key;
    const rows = events.filter(item => !requestId || item.request_id === requestId).slice(0, 60).reverse();
    $("tool-chain-empty").hidden = rows.length > 0;
    const focus = document.activeElement?.dataset.receipt;
    const detailFocus = document.activeElement?.dataset.receiptDetail;
    $("tool-chain").replaceChildren(...rows.map((item, index) => {
      const sameRequest = index > 0 && item.request_id && item.request_id === rows[index - 1].request_id;
      const row = make("li", `tool-dispatch-node ${sameRequest ? "is-linked" : ""} ${item.status === "error" ? "is-error" : ""}`, "");
      const button = make("button", "", ""); button.type = "button"; button.dataset.receipt = item.id;
      const phase = item.event === "tool_call_started" ? "DISPATCH" : item.status === "error" ? "FAILED" : "RETURNED";
      button.append(make("span", "tool-dispatch-phase", `${phase} · ${time(item.at)}`), make("strong", "", item.detail || "Tool"), make("span", "muted", item.tool_round == null ? "Round not retained" : `Round ${item.tool_round}`));
      button.addEventListener("click", () => inspect(item)); row.append(button);
      const disclosure = make("details", "tool-call-details", ""), summary = make("summary", "", "Arguments & output"), content = make("div", "tool-call-body", "Loading call details…");
      summary.dataset.receiptDetail=item.id;
      disclosure.append(summary,content); disclosure.open = expanded.has(item.id);
      disclosure.addEventListener("toggle", () => { if (disclosure.open) { expanded.add(item.id); loadDetails(item,content); } else expanded.delete(item.id); });
      row.append(disclosure); return row;
    }));
    const chosen = rows.find(item => item.id === selected) || rows[rows.length - 1];
    if (chosen) inspect(chosen);
    else { $("tool-receipt-title").textContent = "No retained dispatch"; $("tool-receipt-facts").replaceChildren(); }
    if (focus) [...$("tool-chain").querySelectorAll("button")].find(button => button.dataset.receipt === focus)?.focus({ preventScroll: true });
    if (detailFocus) [...$("tool-chain").querySelectorAll("summary")].find(node => node.dataset.receiptDetail === detailFocus)?.focus({preventScroll:true});
  }
  $("tool-request")?.addEventListener("change", event => { requestId = event.target.value; selected = null; draw(); });
  window.OmniToolActivity = { clear() { generation++; drawKey=""; cache.clear(); expanded.clear(); $("tool-chain")?.querySelectorAll(".tool-call-details").forEach(node => {node.open=false;node.querySelector(".tool-call-body").replaceChildren();}); }, render(snapshot, access) {
    if (!$("tool-chain")) return;
    token = access;
    events = (snapshot.timeline || []).filter(item => ["tool_call_started", "tool_call_completed"].includes(item.event));
    const returns = events.filter(item => item.event === "tool_call_completed");
    $("tool-return-count").textContent = String(returns.length);
    $("tool-error-count").textContent = String(returns.filter(item => item.status === "error").length);
    $("tool-connection-count").textContent = String(snapshot.connections?.servers?.length || 0);
    const latest = events[0], fresh = latest && Date.now() - new Date(latest.at).getTime() < 120000;
    $("tool-signal").textContent = !latest ? "QUIET" : !fresh ? "RETAINED" : latest.event === "tool_call_started" ? "DISPATCH" : latest.status === "error" ? "ERROR" : "RETURNED";
    $("tool-signal-detail").textContent = latest ? `${latest.detail || "Tool"} · ${time(latest.at)} · recorded signal` : "No tool calls in the retained sample";
    const ids = [...new Set(events.map(item => item.request_id).filter(Boolean))], key = JSON.stringify(ids);
    if (optionsKey !== key) {
      if (!ids.includes(requestId)) requestId = "";
      const all = make("option", "", "All retained requests"); all.value = "";
      $("tool-request").replaceChildren(all, ...ids.map(id => { const option = make("option", "", id); option.value = id; return option; }));
      $("tool-request").value = requestId; optionsKey = key;
    }
    $("tool-request").disabled = ids.length === 0;
    draw();
  } };
})();
