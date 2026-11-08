# Durable voice memory and temporal retrieval

Voice exchanges commit to a device-local SQLite source journal before any
embedding work. `history_events` retains the original text, event time, record
time, SHA-256, kind and speaker fields. SQLite uses WAL and FULL synchronous
commits. A busy encoder, denied model admission, full indexing queue or restart
can delay semantic indexing; it cannot discard an already committed event.
Event time is captured when the turn is submitted, before reasoning and TTS.
Decoded user words are retained on interrupted and failed return paths too;
unplayed assistant proposals are omitted from those conversation records.
Indexing retries from the database after resource admission. No new model is
downloaded: semantic ranking keeps the existing configured memory embedder.

The old `memories` table remains a derived semantic cache. Near-duplicate cache
merges and cache decay never delete journal events. Decay now applies only to
the elapsed interval since the last application, avoiding repeated exponential
penalties on every restart. Hour/day `history_buckets` contain extractive views,
source IDs and source hashes. `history_links` ties occurrences to derived
semantic clusters. These views preserve the distinction between raw events and
derived organization; they make no new factual assertions. `rebuild_views()`
reconstructs lexical and temporal views from source events.

## Retrieval contract

`conversation_search` is a server-pinned tool in the memory capability family.
It is authorized only for the local voice session derived from the current
portal token; other cookie sessions cannot read the device archive. Existing
`memory_write`, `memory_read`, `memory_search` and browser Trash/TTL behavior
remain session scoped. Device-owner voice history is a separate durable store.
A daemon token change rebinds the local voice session without changing the
archive file. Explicit alternate voice-session identities do not acquire archive
access automatically.
The authorized owner receives the archive schema on the first reasoning pass,
including after audio comprehension, without having to discover a capability
family. Exposure does not require a lookup on ordinary turns, and explicit tool
allowlists still apply. Other sessions do not receive this archive shortcut.
On bearer-token refresh the default harness cookie is rebound to the current
voice identity; explicitly configured client identities remain unchanged.

The reasoning model expresses temporal intent through typed arguments. There
is no keyword-based memory router. Use an empty `query` for a chronological
period overview, or a topic-only query for semantic retrieval. Select typed
`view=overview` for a recap and `view=search` for a specific fact. An overview
returns dated source excerpts sampled across the beginning, middle and end of
each day, with source IDs/hashes and complete-period-index metadata. It avoids
repeated semantic refinement to assemble a broad recap. `retrieval_complete`
means the bounded candidate search has finished; a result limit is not a search
failure. Overview excerpts remain samples, not a claim to cover every fact. The typed `kind`
defaults to `conversation`, keeping ambient acoustic observations and
unattributed `heard` fragments out of conversation recaps. Records explicitly routed to `SELF` and legacy exchanges of unknown
attribution remain eligible. Completed exchanges explicitly routed to
`AMBIGUOUS` or `OTHER` are excluded too: generating a reply did not establish
who addressed the agent. They remain available with `kind=all` and contribute
to `unattributed_speech_in_window`; filtering never changes the source journal.
`SELF` means addressed to the agent, not a verified human identity. `sound` and `all`
explicitly include non-speech and all retained sources respectively. The result
reports excluded unattributed speech counts, so an empty conversation window
does not assert that nobody spoke. All original records remain searchable with
`kind=all`. Time filtering
precedes both semantic candidate selection and SQLite FTS5 ranking; reciprocal
rank fusion combines those independent retrieval routes. If the encoder is
unavailable or lacks memory headroom, lexical and chronological retrieval remain
available and the result reports the semantic limitation.

The model also declares `after_retrieval=answer` when this lookup should produce
the reply. After one successful, complete archive lookup, the runtime honors
that plan with an answer pass without tools; it cannot keep refining the same
recap indefinitely. Choose `continue` when comparing periods, paging, or using
retrieved information in another tool action. Missing fields retain the legacy
`continue` behavior; failed or incomplete lookups keep their tool chain open.
The reply must identify limited/empty evidence and legacy date uncertainty.
Ambiguous or other-directed recorded exchanges do not establish the current
speaker's identity; relative words in old sources retain their original date.
The answer phase uses standard `tool_choice=none` and retains its policy when
active virtual context rebuilds the prompt from the admission contract.

| Period | Meaning |
| --- | --- |
| `today` | Local midnight through now |
| `yesterday` | Previous local calendar day, including 23/25-hour DST days |
| `last_week` | Previous Monday 00:00 through this Monday 00:00 |
| `last_7_days` | Previous 168 elapsed hours |
| `past_hour` | Previous 60 elapsed minutes |
| `hour_ago` | 30–90 elapsed minutes before now |
| `range` | Explicit offset-aware ISO 8601 start and exclusive end |
| `all` | All retained past events |

The default IANA zone comes from the device configuration; a caller may supply
an explicit IANA `timezone`. Every response includes the resolved range, zone,
interpretation and reference time. Chronological pages return `next_before_id`;
pass it as `before_id` with the same period and `view=search` to continue.
A completed overview does not require event-page traversal. Results carry source
IDs, timestamps and hashes, speaker attribution, truncation flags and historical
trust labels. Assistant output is not a verified user fact or current perception.
The tool defaults to four results, permits eight, and caps evidence payloads.
Semantic scans examine at most 4,096 time-filtered events; the response discloses
that limit. FTS can retrieve matching older records beyond that vector scan.

## Migration and operational limits

Migration copies surviving legacy cache entries once, with stable `legacy:ID`
keys. It preserves text, vectors and original timestamps and explicitly labels
those timestamps uncertain: older consolidation could replace text while keeping
an earlier creation time. Deleted memories, failed old embedding-dependent
writes and exact occurrences already collapsed by the previous implementation
cannot be reconstructed. Existing private backups must remain private.

Before rollout, use SQLite's backup API on the live archive and verify the
backup. Check migrated counts and source hashes after deployment. Validate
storage, retrieval and model answering separately: a direct store check does not
prove that the live model selected the right tool or answered from its evidence.
Do not put private conversation excerpts, portal tokens or session identities
in committed evaluation reports.

## Research basis, reviewed 2026-11-08

Google's [SLEEP overview](https://research.google/blog/the-sleep-framework-a-new-way-for-llms-to-retain-knowledge-in-continual-learning/)
and [Language Models Need Sleep](https://arxiv.org/abs/2606.03979) describe
consolidation into learned parameters and rehearsal. That requires architecture
and training support. This repair takes inspiration from separating immediate
capture from later consolidation; it implements external source-preserving
memory and does not claim SLEEP's parametric learning or benchmark results.

[TSHMem, EMNLP 2026](https://aclanthology.org/2026.emnlp-main.1034/) couples
temporal intent with memory retrieval. The adopted engineering principle is to
apply time constraints during candidate selection and retain source-linked
hierarchies; this implementation does not reproduce its heterogeneous graph.

[RecMem, ACL Findings 2026](https://aclanthology.org/2026.findings-acl.1619/)
separates incoming interactions from later consolidation and recovers detail
from underlying evidence. The archive retains every captured source event while
performing derived work asynchronously; it does not reproduce RecMem's learned
extraction policy.

[EvalMem, EMNLP Findings 2026](https://aclanthology.org/2026.findings-emnlp.1134/)
distinguishes storage, retrieval and generation failures. Our validation follows
that distinction, including offline retention tests, temporal retrieval checks,
and actual model-selected live tool calls. Paper-reported benchmark scores do
not establish reliability on this device; local rollout evidence is required.
