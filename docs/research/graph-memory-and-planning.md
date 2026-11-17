# Source-grounded graph memory and execution

This implementation targets natural encoding and recall across text, speech,
non-speech sound, images, video, tools, sensors and environment observations.
Human equivalence and success on arbitrary tasks are evaluation goals, not
properties established by adding a graph.

## Memory design and research basis

The immutable event journal is authoritative. Model-derived summaries, entities,
aliases and semantic/entity/temporal/causal relations are rebuildable indexes.
Every accepted relation contains exact quotations, source IDs, hashes, original
source times, modality and authority. Unsupported quotations fail atomically.
A historical observation never becomes current perception or current environment
verification. Addressed speech does not establish the speaker's identity.

| Primary source | Adopted idea | Deliberate boundary |
| --- | --- | --- |
| [MAGMA, ACL 2026](https://arxiv.org/abs/2601.03236), revised April 16 | Separate semantic, entity, temporal and causal relationships; commit sources before deferred organization | Relation typing is generated and source-checked; causal truth is not proven by a matching quote |
| [Zep, 2025](https://arxiv.org/abs/2501.13956) | Separate episodes from derived facts; preserve event and ingestion time and fact validity history | Private owner and browser scopes remain independent; reported benchmark gains do not transfer automatically |
| [HippoRAG 2, ICML 2025](https://arxiv.org/abs/2502.14802) | Combine source retrieval, linked entities and personalized PageRank | Local bounded neighborhoods and rank fusion are used; this is not a reproduction of the paper's trained components or benchmark |
| [A-MEM, 2025](https://arxiv.org/abs/2502.12110) | Resolve references and link new experience to existing notes | Derived organization can evolve; immutable source events cannot be overwritten |
| [LongMemEval, ICLR 2025](https://arxiv.org/abs/2410.10813) | Evaluate extraction, multi-session reasoning, temporal recall, updates and abstention separately | Local fixtures are not an official LongMemEval score |

### Encoding and retrieval

Sources commit synchronously before embedding or language inference. Encoding
uses the installed language worker with deterministic structured output,
`think:false`, no tools and no speech generation. Optional background encoding
yields its queue lane to foreground requests and fresh native speech activity.
Long source text is segmented without dropping Unicode characters. Assistant
narration remains searchable context and cannot independently establish external
facts. Optional model failures leave sources available to lexical retrieval.

Time windows use IANA time zones and precede lexical, semantic and graph
candidate selection. Retrieval searches the filtered archive rather than a
recent-only slice. FTS5, installed embeddings, entity aliases and a bounded
source-linked graph neighborhood contribute to rank fusion. The query reports
semantic availability and bounded graph coverage. Current request episodes are
excluded from historical recall.

Explicit same-authority corrections may close an earlier fact's validity;
old source events and facts remain available. Uncertain, mixed or unidentified
speech cannot overwrite a user's established fact. Exact support does not prove
that a derived predicate is a sound interpretation: the inspector and answer
pass must retain epistemic status and authority.

### Scope and retention

Each browser cookie has a private source/graph database. Trash invalidates active
captures and encoding generations and deletes that browser database and WAL.
The device-owner voice archive is separate and survives browser Trash. Existing
voice events are imported from a SQLite backup with integrity and source-hash
verification; incremental replay preserves original timestamps and legacy
uncertainty. Other cookies receive device aggregate structure only.

Location and ephemeral web/document/session-memory payloads are not copied into
the durable event store. Their bounded receipts may be retained without the
protected content. Actual uploaded media bytes and private model reasoning are
never stored in this graph. Generic sensor/event submissions are authenticated
client reports, not forged producer-authoritative perception.

## Planning research and implementation boundary

[Plan-and-Act, ICML 2025](https://proceedings.mlr.press/v267/erdogan25a.html)
separates a high-level planner from an executor and revises the plan as execution
reveals new information. Its planner training and web-navigation evaluation are
material parts of the result. We adopt the architectural separation and require
local execution evidence; prompt changes alone cannot inherit its performance.

[GAP, 2025](https://arxiv.org/abs/2510.25320) represents tool dependencies in a
DAG and schedules independent levels separately. Its supervised and reinforcement
training were evaluated primarily on question-answering workloads. A visual graph
or a sequential dependency chain does not establish equivalent general-purpose
planning ability or authorize parallel state-changing desktop actions.

[Agent Workflow Memory, ICML 2025](https://proceedings.mlr.press/v267/wang25bx.html)
extracts reusable workflows from prior trajectories for later tasks. Reusing a
workflow requires checking its preconditions against the current environment;
a historical success is not proof that a present application, device or protocol
has the same state.

The existing controller already separates knowledge, environment versions,
finite committed frontiers and current verification. A new task must still
inspect uncertainty, execute a concrete capability, assess every criterion,
recover from a failed assumption and verify current effects. Discovery and
successful command exit alone are not progress. Research and shell are available
mechanisms for finding an equivalent route when the first capability fails.

## Validation

`tests/test_episodic_memory.py` covers source durability, all modality enums,
quoted support, validity updates, attribution, graph traversal, temporal filters,
retry and retrieval after more than 8,000 newer events.
`tests/test_episodic_memory_service.py` covers foreground priority, cancellation,
stream capture, scope isolation, Trash races, legacy backup replay, Unicode
segmentation, ephemeral-payload exclusion and the closed retrieval-answer phase.
Actual capture, model encoding, retrieval and model answering require separate
live evidence. Record local model/version, fixtures, outcomes, latency and failures
before making any empirical quality claim.
