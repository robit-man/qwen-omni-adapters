/* Committed plans and executor receipts; no inferred completion or private reasoning. */
(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const make = (tag, cls, text) => { const element = document.createElement(tag); element.className = cls; element.textContent = text; return element; };
  let plans = [], taskId = null, selected = null, optionsKey = "";
  const labels = { planned: "Planned", evidence_acquired: "Evidence acquired", awaiting_audit: "Awaiting verification", verified: "Verified", action_failed: "Action failed", expected_effect_missing: "Effect missing", audit_failed: "Verification failed" };
  function inspect(plan) {
    const step = plan.nodes.find(item => item.id === selected) || plan.nodes.find(item => item.active) || plan.nodes[0];
    if (!step) return;
    selected = step.id;
    $("plan-step-title").textContent = step.subtask || step.decision;
    const fields = [["Tool family", step.family], ["Operation", step.operation], ["Selection", step.selection_reason || "No selection explanation retained"],
      ["Expected effect", step.expected_effect], ["Target", step.target || "Not specified"], ["Acceptance test", step.acceptance || "Not retained"],
      ["Verification", step.verification_family], ["State", labels[step.status] || step.status], ["Source receipts", step.evidence_ids.join(" · ") || "No executor receipt yet"]];
    $("plan-step-facts").replaceChildren(...fields.map(([label, text]) => { const row = document.createElement("div"); row.append(make("dt", "", label), make("dd", "", text)); return row; }));
    $("plan-requirements").replaceChildren(...plan.requirements.map(item => make("li", "", `${item.status || "pending"} · ${item.text}${item.evidence_ids.length ? ` · ${item.evidence_ids.join(", ")}` : ""}`)));
    $("plan-steps").querySelectorAll("button").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.step === selected)));
  }
  function draw() {
    const plan = plans.find(item => item.task_id === taskId) || plans[0];
    $("plan-empty").hidden = Boolean(plan?.nodes.length);
    $("plan-workbench").hidden = !plan?.nodes.length;
    $("plan-empty").textContent = plan ? "The controller has not committed steps yet. Its current phase is shown above." : "Committed steps appear when a task starts. Discovery alone does not count as completion.";
    if (!plan) { $("plan-meta").textContent = "No retained task plan"; return; }
    taskId = plan.task_id;
    $("plan-task").value = taskId;
    $("plan-meta").textContent = `${plan.status} · ${plan.phase || "queued"} → ${plan.next_transition || "waiting"} · revision ${plan.revision} · environment ${plan.environment_version}`;
    const focus = document.activeElement?.dataset.step;
    $("plan-steps").replaceChildren(...plan.nodes.map(step => {
      const row = make("li", `plan-step ${step.active ? "is-current" : ""} ${step.status === "verified" ? "is-verified" : ""}`, "");
      const button = make("button", "", ""); button.type = "button"; button.dataset.step = step.id;
      button.append(make("span", "plan-step-status", labels[step.status] || step.status), make("strong", "", step.subtask || step.decision), make("span", "muted", `${step.decision} · ${step.family} · ${step.operation}`));
      button.addEventListener("click", () => { selected = step.id; inspect(plan); $("plan-announcement").textContent = `Selected ${step.subtask}. Evidence details updated.`; });
      row.append(button); return row;
    }));
    inspect(plan);
    if (focus) [...$("plan-steps").querySelectorAll("button")].find(button => button.dataset.step === focus)?.focus({ preventScroll: true });
  }
  $("plan-task")?.addEventListener("change", event => { taskId = event.target.value; selected = null; draw(); });
  window.OmniTaskPlan = { render(snapshot) {
    if (!$("plan-task")) return;
    plans = snapshot.work?.plans || [];
    $("plan-task").disabled = plans.length === 0;
    const key = JSON.stringify(plans.map(plan => [plan.task_id, plan.objective]));
    if (key !== optionsKey) {
      $("plan-task").replaceChildren(...plans.map(plan => { const option = make("option", "", plan.objective); option.value = plan.task_id; return option; }));
      optionsKey = key;
    }
    draw();
  } };
})();
