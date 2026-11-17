/* Request-linked dispatch receipts. No timing-based or inferred task chains. */
(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const make = (tag, cls, text) => { const node = document.createElement(tag); node.className = cls; node.textContent = text; return node; };
  let events = [], requestId = "", selected = null, optionsKey = "";
  const time = value => value ? new Date(value).toLocaleTimeString() : "Time not retained";
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
    const rows = events.filter(item => !requestId || item.request_id === requestId).slice(0, 60).reverse();
    $("tool-chain-empty").hidden = rows.length > 0;
    const focus = document.activeElement?.dataset.receipt;
    $("tool-chain").replaceChildren(...rows.map((item, index) => {
      const sameRequest = index > 0 && item.request_id && item.request_id === rows[index - 1].request_id;
      const row = make("li", `tool-dispatch-node ${sameRequest ? "is-linked" : ""} ${item.status === "error" ? "is-error" : ""}`, "");
      const button = make("button", "", ""); button.type = "button"; button.dataset.receipt = item.id;
      const phase = item.event === "tool_call_started" ? "DISPATCH" : item.status === "error" ? "FAILED" : "RETURNED";
      button.append(make("span", "tool-dispatch-phase", `${phase} · ${time(item.at)}`), make("strong", "", item.detail || "Tool"), make("span", "muted", item.tool_round == null ? "Round not retained" : `Round ${item.tool_round}`));
      button.addEventListener("click", () => inspect(item)); row.append(button); return row;
    }));
    const chosen = rows.find(item => item.id === selected) || rows[rows.length - 1];
    if (chosen) inspect(chosen);
    else { $("tool-receipt-title").textContent = "No retained dispatch"; $("tool-receipt-facts").replaceChildren(); }
    if (focus) [...$("tool-chain").querySelectorAll("button")].find(button => button.dataset.receipt === focus)?.focus({ preventScroll: true });
  }
  $("tool-request")?.addEventListener("change", event => { requestId = event.target.value; selected = null; draw(); });
  window.OmniToolActivity = { render(snapshot) {
    if (!$("tool-chain")) return;
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
