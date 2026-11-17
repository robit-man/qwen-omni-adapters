(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  let token = "", enabled = false;
  const pending = new Set();
  async function request(path, options = {}) {
    const response = await fetch(path, { ...options, headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" }, cache: "no-store" });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Connection request failed");
    return result;
  }
  const changed = () => window.dispatchEvent(new Event("omni-connections-changed"));
  $("mcp-connect-form")?.addEventListener("submit", async event => {
    event.preventDefault(); if (!enabled) return;
    const form = event.target, fields = new FormData(form), submit = form.querySelector("button");
    const body = { endpoint: String(fields.get("endpoint") || "").trim(), key: String(fields.get("key") || ""), scope: fields.get("scope") };
    submit.disabled = true; form.setAttribute("aria-busy", "true"); $("mcp-message").textContent = "Connecting and discovering advertised tools…";
    try {
      const result = await request("/api/mcp/servers", { method: "POST", body: JSON.stringify(body) });
      $("mcp-message").textContent = `Connected ${result.server.name} · ${result.tools} tools discovered.`;
      form.reset(); changed();
    } catch (error) { $("mcp-message").textContent = error.message; }
    finally { form.elements.key.value = ""; submit.disabled = false; form.removeAttribute("aria-busy"); }
  });
  window.OmniMCPConnections = { render(snapshot, access) {
    if (!$("mcp-server-list")) return;
    token = access; enabled = snapshot.connections?.mcp_enabled === true;
    if (!$("mcp-connect-form").hasAttribute("aria-busy")) $("mcp-connect-form").querySelectorAll("input, select, button").forEach(element => { element.disabled = !enabled; });
    const servers = snapshot.connections?.servers || [];
    $("mcp-count").textContent = enabled ? `· ${servers.length} registered` : "· unavailable";
    const focus = document.activeElement?.dataset.mcpControl;
    $("mcp-server-list").replaceChildren(...servers.map(server => {
      const row = document.createElement("li"), name = document.createElement("span");
      name.textContent = `${server.name} · ${server.transport.toUpperCase()} · ${server.authenticated ? "key configured" : "no key"} · ${server.scope}`; row.append(name);
      for (const action of ["refresh", "remove"]) {
        const button = document.createElement("button"); button.className = "chip"; button.type = "button"; button.textContent = action === "refresh" ? "Refresh" : "Remove"; button.dataset.mcpControl = `${server.id}-${action}`; button.disabled = pending.has(server.id);
        button.setAttribute("aria-label", `${button.textContent} ${server.name}`);
        button.addEventListener("click", async () => {
          pending.add(server.id); button.disabled = true; $("mcp-message").textContent = `${action === "refresh" ? "Refreshing" : "Removing"} ${server.name}…`;
          try { await request(`/api/mcp/servers/${encodeURIComponent(server.id)}`, { method: action === "remove" ? "DELETE" : "POST" }); $("mcp-message").textContent = action === "remove" ? "Connection and credential removed." : "Catalog refreshed."; changed(); }
          catch (error) { $("mcp-message").textContent = error.message; button.disabled = false; }
          finally { pending.delete(server.id); }
        }); row.append(button);
      }
      return row;
    }));
    if (focus) [...$("mcp-server-list").querySelectorAll("button")].find(button => button.dataset.mcpControl === focus)?.focus({ preventScroll: true });
  } };
})();
