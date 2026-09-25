# Agent control and medium-horizon evaluation findings

Status: implementation-driving synthesis

Date: 2026-09-25

Scope: efficient, evidence-grounded control of background coding, browser, and
desktop tasks on the resident Ornith/Qwen Omni Jetson runtime

## Research question

Why can an agent with enough model capability and a lossless long-context layer
still spend many turns inspecting the same state, choose an inappropriate action,
or claim progress without building the requested artifact?

The reviewed evidence points to the agent-computer interface and evaluator, not
only the model or memory size. Long history can preserve a failure perfectly while
still presenting an excessively broad action space. The practical control layer
must expose compact task-relevant actions, divide a task into verified phases, and
score external state rather than prose.

## Corpus and provenance

The full PDFs and extracted text for REF-021 through REF-027 are stored under the
local research source tree. SHA-256 digests are recorded in `source-manifest.json`.
This synthesis was written from the acquired papers, including methods, ablations,
error analyses, and appendices; it is not based only on abstract pages.

Quality shorthand is GRADE-like:

- High: peer-reviewed, executable benchmark or controlled ablation directly
  relevant to the tested behavior.
- Moderate: peer-reviewed or broad experimental work with some transfer risk.
- Low: preprint evidence or a result coupled to a materially different model.

## REF-021 — SWE-agent

Source: [SWE-agent](https://arxiv.org/abs/2405.15793)

Quality: moderate-to-high for agent-computer-interface design; lower for direct
transfer from repository repair to greenfield application construction.

The paper treats the interface between model and computer as an independent
engineering object. Its four useful principles are simple actions, compact actions
that accomplish meaningful work, concise but informative feedback, and guardrails
that stop mistakes from cascading.

The ablations are directly relevant to the live Egg failure. Shell-only navigation
encourages chains of `cd`, `ls`, and `cat`. An iterative result-at-a-time search
interface caused models to exhaustively page through matches, performing worse than
no search. Summarized search with bounded context worked better. Full files and full
trajectory history also underperformed bounded views.

Its edit primitive combines mutation, immediate post-edit context, and syntax
validation. This reduces the number of state transitions and prevents a bad edit
from becoming the premise for later work. The paper reports that removing the edit
tool or lint guard substantially lowers task resolution.

Implementation consequence:

- Keep unrestricted shell as an escape hatch, not the preferred file interface.
- Add compact read/write/patch/list operations with immediate bounded receipts.
- Reject a repeated causal failure even when command spelling changes.
- Provide syntax or parse feedback in the same edit receipt.
- Retain recent high-signal observations, not every raw command output.

## REF-022 — ToolSandbox

Source: [ToolSandbox](https://arxiv.org/abs/2408.04682)

Quality: high for stateful tool evaluation and intermediate-milestone scoring.

ToolSandbox evaluates arbitrary valid trajectories against a directed acyclic graph
of state milestones and minefields. It snapshots databases and tool traces after
turns, then finds the best chronologically valid mapping between trajectory state
and required milestones. This is stronger than matching one prescribed action
sequence and much harder to game with narration.

The execution context separates agent-visible conversation from authoritative
world state. Stateful tools raise informative exceptions when prerequisites are
missing. The paper also deliberately introduces related distractor tools and schema
degradation to measure whether agents can select and use tools rather than memorize
names.

Implementation consequence:

- Represent a medium-horizon build as a milestone DAG, not a prose checklist.
- Score immutable filesystem, process, database, HTTP, and rendered-UI snapshots.
- Add minefields such as modifying an existing project, cross-tenant leakage,
  fabricated citations, missing password hashing, or claiming success after a
  failed test.
- Record partial milestone coverage for diagnosis without turning it into task
  success.
- Keep the evaluator outside the model prompt and never expose expected answers.

## REF-023 — tau-bench

Source: [tau-bench](https://arxiv.org/abs/2406.12045)

Quality: moderate-to-high for end-state evaluation and reliability measurement;
the domains are customer service rather than code construction.

Tau-bench models the interaction as a partially observed process with dynamic user,
agent, tool, and policy state. Its evaluator compares the final database against the
annotated goal state instead of requiring a canonical transcript.

The paper's `pass^k` metric is important here. A model that passes once but fails
under repeated trials is not reliable enough for autonomous deployment. Reported
performance drops sharply when consistency across multiple runs is required. Common
failures include wrong arguments, policy mistakes, and partial resolution of compound
requests.

Implementation consequence:

- Run the same task shape with sealed randomized values and at least three seeds.
- Require all compound criteria, not an average that hides one missing capability.
- Report pass@1 and pass^k alongside latency, actions, retries, and compactions.
- Preserve natural alternative solutions by checking state, not command strings.

## REF-024 — OSWorld

Source: [OSWorld](https://arxiv.org/abs/2404.07972)

Quality: high for execution-based desktop evaluation and observed GUI failure modes.

OSWorld supplies reproducible initial machine states and custom execution-based
evaluators for cross-application tasks. Its analysis separates planning from GUI
grounding: an agent may describe the right action yet miss its coordinates. More
than three quarters of sampled failures involved click inaccuracy, which then caused
repetitive clicks, popups, and state drift.

The paper finds accessibility-tree observations much more useful than screenshots
alone for many tasks, but single trees can consume thousands of tokens. Pure
screenshot trajectory history did not gain from extra rounds. Agents were also
fragile to window moves, resizes, and desktop clutter.

Implementation consequence:

- Use the scoped application window, not the whole desktop, by default.
- Fuse the current screenshot with a pruned accessibility/DOM tree.
- Ground on stable element IDs or current bounding boxes and revalidate before use.
- Retain semantic action receipts; expire coordinates and old screenshots.
- Add window move/resize/clutter perturbations and execution-state scoring to the
  desktop suite.

## REF-025 — Agent S

Source: [Agent S](https://arxiv.org/abs/2410.08164)

Quality: low-to-moderate because it is a preprint and uses a larger hosted model;
moderate confidence in its hierarchical control pattern.

Agent S divides a task into a manager's topologically ordered subtasks and workers'
single grounded actions. Each worker emits status of the previous action, current
observation analysis, a semantic next action, and one grounded action. A DONE signal
advances a subtask; FAIL returns control to the manager for replanning.

Its agent-computer interface combines screenshot and accessibility information, adds
OCR nodes missing from the accessibility tree, assigns element IDs, and exposes one
bounded action per step. It explicitly avoids arbitrary multi-action code for GUI
control because that prevents timely feedback.

The useful distinction is between narrative memory for whole-task strategy and
episodic memory for successful subtask execution. Failed and successful histories
are summarized differently rather than replayed indiscriminately.

Implementation consequence:

- Pin one current subgoal with an explicit exit condition and allowed action class.
- Permit one concrete action, observe, then update subgoal state.
- Replan at a failed subgoal instead of mutating the same action indefinitely.
- Keep the full objective and completion graph authoritative outside derived memory.

## REF-026 — SWE-Bench Pro

Source: [SWE-Bench Pro](https://arxiv.org/abs/2509.16941)

Quality: low-to-moderate because it is a recent preprint; high relevance to realistic
multi-file, long-horizon work.

SWE-Bench Pro contains enterprise-scale issues that may require hours or days for a
professional engineer. It augments problem statements with human-verified requirements
and interface specifications. Removing those augmentations substantially lowers agent
performance, showing that persistent exact contracts matter.

Its trajectory taxonomy distinguishes semantic misunderstanding, incomplete fixes,
wrong files, tool misuse, syntax failures, and context overflow. Those classes should
remain distinct in our telemetry because each calls for a different repair.

Implementation consequence:

- Pin exact paths, public interfaces, and completion criteria independently of chat.
- Evaluate cross-file work, tests, and runtime behavior rather than patch presence.
- Attribute failure to a stable taxonomy instead of a generic stall counter.
- Include at least one multi-file change and one post-build update in each SaaS run.

## REF-027 — BrowserGym

Source: [BrowserGym](https://arxiv.org/abs/2412.05467)

Quality: moderate-to-high for standardized browser observations, actions, and trace
analysis.

BrowserGym standardizes task goals, chat history, screenshots, DOM/accessibility
observations, and browser actions across multiple benchmarks. It supports restricted
high-level action sets as well as raw code, but treats the latter as a research option
with safety and control costs.

Its prompt builder dynamically shrinks history to fit the model budget. AgentXRay
retains step-by-step decisions and observations for diagnosis. The error taxonomy
separates navigation, form handling, task understanding, information extraction, and
environment failures.

Implementation consequence:

- Make scoped observation and action representations explicit and versioned.
- Dynamically budget current observation, task contract, subgoal, and recent receipts.
- Preserve full traces externally while injecting only the active working set.
- Test form controls, nested widgets, downloads, challenges, and visual-only targets
  under the same execution-based harness.

## Cross-paper design

The immediate controller should use a phase contract:

1. `phase_id` and `subgoal` describe one bounded outcome.
2. `entry_evidence` records authoritative state at phase start.
3. `allowed_capabilities` exposes only the relevant compact actions.
4. `exit_predicate` is evaluated against external state.
5. `minefields` name invariant violations that fail the phase.
6. `action_budget` forces checkpoint or manager replanning.
7. `last_causal_outcome` survives compaction and worker restart.

For the SaaS benchmark, a generic milestone DAG is:

1. target root exists and is isolated;
2. three fetched sources are preserved with exact URLs;
3. research and plan documents exist, in that order;
4. application files parse and database schema initializes;
5. authentication and tenant-isolation tests pass;
6. workflow and restart-persistence tests pass;
7. a loopback process serves the application;
8. a browser performs the workflow and a fresh rendered frame is inspected;
9. the final evaluator rechecks all prior milestones and minefields.

These milestones describe external state. They do not prescribe the implementation,
commands, filenames beyond explicit user requirements, or visual design details not
in the request. This prevents reward hacking while allowing valid alternative paths.

## Immediate implementation order

1. Preserve causal failure identity across cosmetic argument changes and restarts.
2. Add compact filesystem actions with bounded post-action receipts and parse checks.
3. Add a deterministic phase ledger derived from explicit criteria and tool evidence.
4. Scope available actions to the current phase while retaining an escape hatch.
5. Evaluate the existing clean run and retain every failure before changing prompts.
6. Rerun with sealed target paths and synthetic values; compare actions and latency.
7. Add browser and desktop perturbation suites only after file/build behavior passes.

## Non-claims

The papers do not prove that hierarchical prompts alone make a 9B model a reliable
software engineer. Agent S results use a different model and environment. SWE-agent
primarily evaluates repository repair, not greenfield SaaS delivery. OSWorld and
BrowserGym show that structured observations help but do not eliminate grounding
errors. ToolSandbox and tau-bench provide evaluator patterns, not a trained policy.

Therefore the design remains model-agnostic and evidence-first. Any improvement must
survive sealed reruns, exact state checks, and negative controls before it becomes a
default runtime policy.
