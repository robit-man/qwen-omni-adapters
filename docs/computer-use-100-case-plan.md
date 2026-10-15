# 100 live desktop acceptance cases

This is a test specification created on 2026-10-15, **not a report of 100
successful runs**. There are 100 distinct scenarios. Run every scenario with
both Qwen and Ornith for **200 model-specific results**. The first ten have
existing application runners; 87 more need orchestration/oracle work and three
require a physical microphone operator. All 200 results start **unrun**.

The machine-readable source is
[computer-use-100-cases.json](../.aiwg/testing/computer-use-100-cases.json).
Each case includes setup, the actual spoken instruction, its independent
success check, required method evidence, automation readiness and time limits.
Repeated Calculator arithmetic deliberately tests two different execution
methods: direct finite targeting and active Kev selection.

GPU scope for this work is **host GPU index 1**, A100 80 GB, UUID
`GPU-170a99ee-850f-2182-1050-4e8d3c87b6b0`, PCI `00000000:82:00.0`, serial
`1323222067591`. Use the UUID, not CUDA ordinal 1: inside a one-GPU child it
normally becomes CUDA device 0. Qwen and Ornith run sequentially. Their
comprehension/language trunk, TTS, pinned Moondream 3.1 and pinned Kev all use
exactly this reserved UUID. Do not start a CPU fallback or use another card.
Optional unrelated decision models remain disabled for this acceptance lane.

Preflight on this date found broker health admission open, no warnings or
active leases, all GPU recovery actions `None`, and the fault recorder active.
The operator's **persistent inference hold remains present**. It is not
removed by this plan; metadata and exact-scope lease interfaces remain available
for separately authorized external tests. Do not broadly resume unrelated
Ollama requests to run this lane. No workload was launched to prepare the plan.

Every normal case uses the selected Omni stack's actual **TTS → ASR →
comprehension → native GUI actions → fresh verification**. Only audio carries
the task into the model; no typed transcript or host-written task plan replaces
it. Mic cases retain actual microphone/VAD evidence and remain distinct from
direct WAV submission. All windows, files and temporary bookmarks belong to
the case; normal default-profile Firefox is used, with existing tabs preserved.
`{marker}` is a fresh session-owned random marker; Workbench is the fresh
case-owned scratch folder announced through the audio/setup.

A click needs physical pointer/button receipts; an accessibility activation
does not prove mouse use. Visual cases need current-frame pinned Moondream
receipts. Kev cases need actual active-policy distributions with the unchanged
0.95 native-probability threshold. A positive task that abstains is incomplete;
only an explicitly designed negative case can pass by verified abstention.
Final outcome, method fidelity, transcript fidelity and conversational completion
are reported separately. Only all required criteria establish a pass.

Unavailable apps, unavailable accessibility and unimplemented orchestration are
explicit blocked/unrun results. Stop the current lane at its first failure,
retain partial artifacts, and diagnose before a fresh selected-case repeat.
Stop owned CUDA workloads on a new GPU fault; do not reset/reboot/release the
hold automatically. Preserve the existing fault recorder and unrelated services.

The tables give concrete goals and fresh oracles. **E** means an existing real
runner; **P** means a specified case needing orchestration; **M** means a case
requiring physical capture/operator coordination. These are readiness labels,
not execution results.


## Previously exposed failures and existing real-app runners

| ID / readiness | Setup and spoken task | Independent success check | Required method |
| --- | --- | --- | --- |
| CU-001 / E | **Select the entire document**. Setup: Gedit with alpha copper ready and one newline. Say: “In the visible Text Editor, select all text in the document and verify the selection. Leave the document content unchanged.” | Fresh AT-SPI selection covers the initial buffer; buffer and saved file bytes are unchanged | keyboard |
| CU-002 / E | **Double-click one visible word**. Setup: Gedit with alpha copper ready and one newline. Say: “In the visible Text Editor, double click the word alpha at the beginning of the document. Verify that exactly the word alpha is selected. Leave all document content unchanged.” | Actual Moondream-bound double-click selects precisely alpha at offsets 0..5; file unchanged | pointer, vision |
| CU-003 / E | **Replace text and click Save**. Setup: Gedit with a fresh speech-note.txt. Say: “In the visible Text Editor, replace the document with the lowercase words copper finch ready, followed by exactly one newline. Click the Save button and verify the saved document.” | Fresh buffer and independently read file both equal copper finch ready plus exactly one newline; Save has a pointer dispatch receipt | keyboard, pointer |
| CU-004 / E | **Inspect and dismiss a context menu**. Setup: Gedit with a fresh speech-note.txt. Say: “In the visible Text Editor, right click inside the document to open its context menu. Inspect the menu, then press Escape to close it. Leave the document content unchanged.” | Right-click exposes a native menu; Escape closes that menu; buffer and disk bytes remain unchanged | pointer, keyboard |
| CU-005 / E | **Complete the spelled-word edit workflow**. Setup: Gedit with a fresh speech-note.txt. Say: “In the visible Text Editor, select all text in the document. Replace it with one line of lowercase words copper finch ready. The last word is spelled R E A D Y. Do not add a blank line or an extra Return. Physically click the Save button using the mouse. Then right click inside the document to open its context menu, inspect that menu, and press Escape to close it. Verify the saved document and closed menu.” | Exact saved bytes, real Save click, observed right-click menu and fresh menu-dismissal verification all hold | pointer, keyboard |
| CU-006 / E | **Calculate using physical button clicks**. Setup: Fresh normal Calculator window. Say: “Use the visible Calculator. Clear the current calculation by clicking its clear button. Calculate seventeen plus twenty five by clicking the number and plus buttons. Click equals and verify the displayed answer.” | Fresh displayed result is 42 after the actual Clear, 17, Plus, 25, Equals inputs; host code never calculates into the app | pointer |
| CU-007 / E | **Find Equals visually**. Setup: Fresh normal Calculator window. Say: “Use the visible Calculator. Clear the current calculation, then enter eight plus seven by clicking its buttons. Finally click the bottom right button with the equals sign, locating it by its visual position and appearance. Verify the displayed answer.” | Result is 15; the Equals dispatch binds the pinned Moondream receipt to its current frame and actual pointer input | pointer, vision |
| CU-008 / E | **Choose Calculator buttons through active Kev**. Setup: Fresh Calculator; active pinned Kev endpoint. Say: “Use the visible Calculator. Clear the current calculation by clicking its clear button. Calculate seventeen plus twenty five by clicking the number and plus buttons. Click equals and verify the displayed answer.” | Result is 42 and accepted native-probability Kev choices are bound to the executed candidates; abstention fails task completion | pointer, kev |
| CU-009 / E | **Complete selection, rename and drag in Files**. Setup: Owned Workbench containing alpha.txt, beta.txt and Destination. Say: “In Files, double click the Workbench folder. Click alpha dot txt once and verify it is selected. Then select alpha dot txt and beta dot txt together and verify both are selected. Right click alpha dot txt and use Rename from its context menu to rename it copper dot txt. Drag copper dot txt onto the Destination folder. Double click Destination and verify copper dot txt is inside. Leave beta dot txt in Workbench and preserve both file contents.” | Single and multiple selection observed; copper.txt ends in Destination; beta.txt remains in Workbench; both file hashes preserved | pointer, keyboard |
| CU-010 / E | **Browse real documentation and recall its source**. Setup: New default-profile Firefox window; fresh session marker. Say: “Use the actual Firefox window that was just opened. Go to Python's official documentation website, open its tutorial, and open the page titled An Informal Introduction to Python. Use mouse clicks to navigate. Use Firefox's Find in page to find Numbers. Scroll the page down with the mouse and read the sentence that describes how the interpreter acts as a simple calculator. Remember that exact sentence together with the actual source address in session memory under the marker {marker}. Leave that sentence visible in Firefox and tell me what you verified. Second spoken turn: Recall the note under {marker}, with its quote and source address, without browsing again.” | Fresh official Python URL, visible acquired sentence, real scroll/click, sourced memory write and second spoken memory-search recall all verified | pointer, keyboard, memory, multi_turn |

## Pointer, selection, clipboard and keyboard behavior

| ID / readiness | Setup and spoken task | Independent success check | Required method |
| --- | --- | --- | --- |
| CU-011 / P | **Single-click without opening**. Setup: Files showing one owned alpha.txt. Say: “Click alpha dot txt once and leave it selected without opening it.” | Exactly alpha.txt is selected; no editor opens and no file bytes change | pointer |
| CU-012 / P | **Drag-select a text phrase**. Setup: Gedit displaying alpha copper ready. Say: “Drag across just the words copper ready and leave them selected.” | Actual button-down/motion/button-up occurs; independent selection is offsets 6..18 with exact selected text | pointer |
| CU-013 / P | **Extend a selection with Shift**. Setup: Gedit displaying alpha copper ready. Say: “Select alpha and then extend the selection to include copper using Shift and the arrow keys.” | Fresh selected range is exactly alpha copper; modifier key receipts exist; file unchanged | keyboard |
| CU-014 / P | **Select nonadjacent files with Control**. Setup: Files showing alpha.txt, beta.txt, gamma.txt. Say: “Select alpha dot txt and gamma dot txt together using Control, leaving beta dot txt unselected.” | Fresh selected-node set is exactly alpha.txt and gamma.txt, with physical modifier-assisted inputs | pointer, keyboard |
| CU-015 / P | **Select a contiguous range with Shift**. Setup: Files in name-sorted list view showing five owned files. Say: “Click the first file and Shift-click the third file so only the first three are selected.” | Exactly first through third are selected in observed order; fourth and fifth remain unselected | pointer, keyboard |
| CU-016 / P | **Copy and paste between owned documents**. Setup: Two Gedit documents, source alpha copper ready and empty destination. Say: “Copy copper from the first document and paste it into the empty second document, without altering the first.” | Actual copy/paste shortcuts occur; destination buffer equals copper and source bytes remain unchanged | keyboard |
| CU-017 / P | **Cut and undo**. Setup: Gedit displaying alpha copper ready. Say: “Cut the word copper, verify it disappeared, then undo and verify the original sentence is restored.” | Fresh intermediate buffer lacks copper; later buffer exactly matches initial bytes; two separately observed states | keyboard, multi_turn |
| CU-018 / P | **Redo a restored edit**. Setup: Gedit with one initial line alpha. Say: “Append beta after alpha, undo that edit, then redo it and leave alpha beta visible.” | Independent snapshots show alpha beta, alpha, alpha beta in order; no disk write required | keyboard |
| CU-019 / P | **Type punctuation literally**. Setup: Empty Gedit document. Say: “Type exactly: Copper, finch! Then add a newline and type: Ready?” | Fresh buffer equals Copper, finch! newline Ready?; no shell, setter or clipboard substitute for required keyboard typing | keyboard |
| CU-020 / P | **Replace a selected middle word**. Setup: Gedit displaying alpha copper ready. Say: “Select only copper and type silver in its place, leaving the other words unchanged.” | Fresh buffer equals alpha silver ready; preserved prefix/suffix and actual keyboard dispatch verified | pointer, keyboard |

## Native application controls and window management

| ID / readiness | Setup and spoken task | Independent success check | Required method |
| --- | --- | --- | --- |
| CU-021 / P | **Switch between owned application windows**. Setup: Owned Gedit and Calculator windows, both initially visible. Say: “Bring Calculator to the front, then return to the Text Editor.” | Fresh foreground XID/PID sequence matches the two owned windows; input never lands in another app | pointer, keyboard |
| CU-022 / P | **Move and resize a window**. Setup: Owned Gedit with recorded initial geometry. Say: “Move this Text Editor toward the left side of the screen and make it smaller without hiding its document.” | Actual pointer drag receipts and independent X11 geometry changes; document remains visible and unchanged | pointer |
| CU-023 / P | **Minimize and restore**. Setup: Owned Calculator with result 42. Say: “Minimize Calculator, then restore it from the desktop and verify forty two is still displayed.” | Window visibility changes and returns; restored foreground identity and result 42 independently verified | pointer |
| CU-024 / P | **Maximize and restore geometry**. Setup: Owned Gedit with recorded normal geometry. Say: “Maximize the Text Editor and then restore its original window size.” | Maximize state appears then clears; restored geometry within compositor tolerance; buffer unchanged | pointer |
| CU-025 / P | **Use a native Save As dialog**. Setup: Unsaved owned Gedit document containing copper finch ready. Say: “Save this document as copper note dot txt in the Workbench folder using Save As.” | Native dialog is observed and dismissed by Save; new owned file has exact expected bytes and path | pointer, keyboard |
| CU-026 / P | **Cancel a native file chooser**. Setup: Owned Gedit and an unused target path. Say: “Open Save As, type draft dot txt as the filename, then cancel without saving.” | File chooser opens and closes; draft.txt does not exist; original file and buffer remain unchanged | pointer, keyboard |
| CU-027 / P | **Toggle a preference and restore it**. Setup: Owned Gedit preferences; original line-number setting recorded. Say: “Turn line numbers on, verify they appear, and then put that setting back how it was.” | Fresh checkbox/property and editor gutter evidence show temporary toggle and exact original value restored | pointer |
| CU-028 / P | **Choose from a native dropdown**. Setup: Owned Gedit preferences with visible tab-width selector and original value recorded. Say: “Set the tab width to four, verify it, then restore the original tab width.” | Dropdown selection and effective property both change to 4 and return to preimage | pointer, keyboard |
| CU-029 / P | **Adjust a media slider**. Setup: Owned media player with a short generated local tone; initial volume recorded. Say: “Set this player's volume near one quarter, pause playback, then restore its original volume.” | Fresh slider value within documented tolerance, pause state verified, original value restored; system volume untouched | pointer |
| CU-030 / P | **Use a spreadsheet through its GUI**. Setup: Owned LibreOffice Calc blank scratch workbook. Say: “Put twelve in A1 and thirty in A2, enter a sum formula in A3, and save the workbook as totals dot ods in Workbench.” | Fresh A3 displays 42; independently opened saved ODS contains the two numbers and formula, not just a screenshot | pointer, keyboard |

## File-manager operations in an owned scratch tree

| ID / readiness | Setup and spoken task | Independent success check | Required method |
| --- | --- | --- | --- |
| CU-031 / P | **Create a folder**. Setup: Files opened to empty owned Workbench. Say: “Create a folder named Copper Notes in this Workbench.” | Native creation dialog and resulting directory observed; independent tree receipt contains exactly Copper Notes | pointer, keyboard |
| CU-032 / P | **Rename a file**. Setup: Files showing owned alpha.txt with recorded hash. Say: “Rename alpha dot txt to meeting notes dot txt using the file manager.” | Old path absent, new path present, file hash unchanged; native rename interaction observed | pointer, keyboard |
| CU-033 / P | **Copy a file to another folder**. Setup: Owned Workbench with alpha.txt and Destination. Say: “Copy alpha dot txt into Destination, leaving the original in Workbench.” | Both paths exist with the original hash; actual GUI copy/paste observed | pointer, keyboard |
| CU-034 / P | **Move a file by cut and paste**. Setup: Owned Workbench with beta.txt and Destination. Say: “Cut beta dot txt and paste it into Destination.” | Source absent, destination hash equals preimage; native cut/paste sequence observed | pointer, keyboard |
| CU-035 / P | **Drag a file onto a folder**. Setup: Owned Workbench with gamma.txt and Destination. Say: “Drag gamma dot txt onto Destination, then open Destination to verify the file is there.” | Real guarded pointer drag and drop; source absent and destination hash unchanged; fresh destination view | pointer |
| CU-036 / P | **Multi-select and copy two files**. Setup: Owned Workbench with alpha.txt, beta.txt and Destination. Say: “Select alpha dot txt and beta dot txt together and copy both into Destination.” | Exactly both selected before copy; four resulting files match preimage hashes; no other file copied | pointer, keyboard |
| CU-037 / P | **Search within an owned folder**. Setup: Owned Workbench containing many files including copper-report.txt. Say: “Search this folder for copper report, then select copper report dot txt from the results.” | Search field and result row observed; selected file resolves to the exact owned path; unrelated files unchanged | pointer, keyboard |
| CU-038 / P | **Sort and change the view**. Setup: Owned Workbench with files of deliberately different names and sizes. Say: “Switch Files to list view, sort by size, then return to name order.” | Fresh native view state and row ordering reflect both sorts; filesystem hashes unchanged | pointer |
| CU-039 / P | **Trash and restore an owned file**. Setup: Owned disposable restore-me.txt; original path/hash recorded. Say: “Move restore me dot txt to Trash, then restore that same file to its original folder.” | Exact owned item enters Trash then returns with original hash; no preexisting Trash item touched | pointer, keyboard |
| CU-040 / P | **Handle a name collision by cancelling**. Setup: Owned alpha.txt and beta.txt with different hashes. Say: “Try renaming alpha dot txt to beta dot txt; when the conflict appears, cancel and keep both files.” | Native conflict is observed; cancellation leaves both names and original hashes intact; no overwrite | pointer, keyboard |

## Actual default-profile Firefox interaction

| ID / readiness | Setup and spoken task | Independent success check | Required method |
| --- | --- | --- | --- |
| CU-041 / P | **Type a URL with real keyboard input**. Setup: New owned default-profile Firefox about:blank window. Say: “Go to https://docs.python.org/3/ by typing it in the address bar and pressing Enter.” | Guarded type_text and Enter dispatch; fresh actual address and document identify official Python docs, not address-bar text alone | keyboard |
| CU-042 / P | **Follow a visible documentation link**. Setup: Owned Firefox at official Python docs home. Say: “Click the Tutorial link and verify the tutorial page opened.” | Actual link click plus fresh committed document URL and tutorial heading | pointer |
| CU-043 / P | **Back and forward history**. Setup: Owned Firefox with docs home then tutorial in history. Say: “Go back to the documentation home and then forward to the tutorial again.” | Two fresh document identities and corresponding current addresses in order | pointer, keyboard |
| CU-044 / P | **Open and switch real tabs**. Setup: Owned Firefox window with Python docs open. Say: “Open Git's official documentation in another tab, switch back to Python, then return to the Git tab.” | Exactly one additional owned tab; active-tab URL sequence Python, Git; no isolated browser profile | pointer, keyboard |
| CU-045 / P | **Close only the newly opened tab**. Setup: Owned Firefox with baseline tab inventory and one test Git tab. Say: “Close the Git tab we just opened and leave the earlier Python tab alone.” | Test tab removed, baseline tabs preserved, Python active; tab close dispatch recorded | pointer, keyboard |
| CU-046 / P | **Find a phrase in a page**. Setup: Owned Firefox on Python introduction page. Say: “Find the words floor division in this page and leave the matching passage visible.” | Find field has floor division; actual match is selected or highlighted and visible in read-only document content | keyboard |
| CU-047 / P | **Scroll to a section and return**. Setup: Owned Firefox on Python introduction page at top. Say: “Scroll down to the Text section, then return to the top of the page.” | Real wheel inputs, changed viewport, section visible at intermediate state, top heading visible at end | pointer |
| CU-048 / P | **Use page zoom and restore it**. Setup: Owned Firefox with original zoom recorded. Say: “Increase this page's zoom to one hundred twenty five percent, then restore the original zoom.” | Fresh browser zoom state changes to 125 and returns to preimage; document URL unchanged | keyboard, pointer |
| CU-049 / P | **Use a browser context menu**. Setup: Owned Firefox over a Python tutorial link. Say: “Right click the Tutorial link, inspect its menu, then dismiss the menu without navigating.” | Real right-click exposes link-specific native menu; Escape closes it; current document and tab count unchanged | pointer, keyboard |
| CU-050 / P | **Bookmark and remove the test bookmark**. Setup: Owned Firefox at Python introduction; exact bookmark preimage recorded. Say: “Bookmark this page as Copper Test Bookmark, verify it exists, then remove only that bookmark.” | Newly created bookmark has exact current URL and title; removal restores preimage and preserves existing bookmarks | pointer, keyboard |

## Real website and documentation exploration

| ID / readiness | Setup and spoken task | Independent success check | Required method |
| --- | --- | --- | --- |
| CU-051 / P | **Read a Python arithmetic explanation**. Setup: Firefox at Python docs home. Say: “Use the tutorial to find how Python does floor division, and explain the operator after reading it on the page.” | Native page navigation and fresh visible source establish the operator; answer agrees with that source and includes its URL | pointer, keyboard |
| CU-052 / P | **Follow an in-page anchor**. Setup: Firefox on Python introduction page. Say: “Use the page's contents to jump to Lists, and read the first visible example there.” | Actual contents-link click changes anchor/viewport; Lists heading and example visible in fresh document | pointer |
| CU-053 / P | **Navigate a documentation index**. Setup: Firefox at Python docs home. Say: “Find pathlib in the library reference and open its documentation.” | Click-driven index traversal reaches official library/pathlib.html and its module heading | pointer, keyboard |
| CU-054 / P | **Use a documentation search field**. Setup: Firefox at Python docs home. Say: “Use the website's search field to find JSON encoding and open the json library reference.” | Actual search input/results/link navigation; committed library/json.html document with read-only module content | pointer, keyboard |
| CU-055 / P | **Explore MDN button documentation**. Setup: Firefox in owned new blank tab. Say: “Open MDN's HTML button reference and find what its disabled attribute does.” | Official MDN current URL and visible attribute explanation acquired through GUI; answer cites visible source | pointer, keyboard |
| CU-056 / P | **Follow a cross-reference**. Setup: Firefox on MDN HTML button reference. Say: “Follow the accessibility-related link from this page and tell me which documentation page you reached.” | Actual visible-link click; fresh target document heading and actual address agree with final answer | pointer |
| CU-057 / P | **Navigate Git command documentation**. Setup: Firefox at Git's official documentation home. Say: “Open git status documentation, find its short format option, and describe it using the page.” | Fresh git-status page, visible short-format option text and matching source URL; no shell execution | pointer, keyboard |
| CU-058 / P | **Compare two real documentation tabs**. Setup: Owned Firefox tabs at Python pathlib and json references. Say: “Read both pages and tell me which one is for filesystem paths and which one is for JSON, naming each source.” | Fresh observed content from both tabs; attribution matches each actual URL; current and historical observations remain distinct | pointer, keyboard |
| CU-059 / P | **Select a source passage visibly**. Setup: Firefox on Python introduction at Numbers. Say: “Use the mouse to select the short sentence describing the interpreter as a calculator, leaving it selected.” | Actual pointer text-selection sequence and fresh document selection agree with the visible source substring | pointer |
| CU-060 / P | **Recover from a missing documentation page**. Setup: Firefox on official Python docs. Say: “Try the nonexistent page https://docs.python.org/3/copper-test-missing.html, then recover using the visible page or browser controls and open the tutorial.” | Failure page observed and attributed; subsequent committed tutorial document verified; no fabricated successful first navigation | pointer, keyboard, negative |

## Observed-source memory and multi-turn reuse

| ID / readiness | Setup and spoken task | Independent success check | Required method |
| --- | --- | --- | --- |
| CU-061 / P | **Store a sourced documentation note**. Setup: Firefox on Python introduction; empty session memory. Say: “Read one short Numbers fact and remember it under marker {marker}, including the current source address.” | Model memory_write stores marker, actual visible quote/fact and canonical current URL; readback matches acquired source | pointer, memory |
| CU-062 / P | **Recall after leaving the source**. Setup: Same session after a sourced note; Firefox changed to unrelated blank tab. Say: “Without browsing again, recall the fact stored under {marker} and its source address.” | Actual memory_search returns same note; answer agrees; no new page navigation or network search during recall | memory, multi_turn |
| CU-063 / P | **Correct an earlier memory note**. Setup: Same session with marker containing a deliberately mistaken draft note. Say: “Reopen the source for {marker}, check it, and replace the incorrect note with what the page actually says.” | Fresh page acquisition precedes memory update; current note corrected with source; old draft never presented as verified | pointer, memory, multi_turn |
| CU-064 / P | **Recall an action recipe and apply it**. Setup: Session contains a source-verified GUI recipe from an earlier successful owned-file rename. Say: “Recall how you renamed a file earlier and use that approach to rename delta dot txt to copper dot txt here.” | Actual memory_search plus fresh UI resolution; new file rename/hash receipt; no reuse of earlier opaque action IDs | pointer, keyboard, memory, multi_turn |
| CU-065 / P | **Separate two similar notes**. Setup: Same session with independently sourced Python and Git notes under distinct markers. Say: “Recall the Python note under {marker}, not the Git note, and give its source.” | Search and final use the requested marker/source; Git content is not substituted | memory, multi_turn |
| CU-066 / P | **Keep another session's note private**. Setup: Session A has a random private marker; session B is fresh. Say: “Look in this session's memory for {marker}; if nothing is there, say you could not find it.” | Session B search contains no A note/source or private canary; final accurately reports absence, with no mutation | memory, negative |
| CU-067 / P | **Trash clears remembered task state**. Setup: Owned portal session with a canary note; only that session is trashed between speech turns. Say: “Recall the note under {marker} from this session.” | Post-Trash memory_search is empty and prior action IDs invalid; unrelated session remains intact | memory, multi_turn, negative |
| CU-068 / P | **Treat a recalled page as historical**. Setup: Session has a sourced Python note; current Firefox page is Git. Say: “Recall the Python fact under {marker}, and tell me separately what page is currently open.” | Stored note labeled recalled/historical; current claim uses fresh Git observation; provenance never recasts recalled text as current vision | memory, multi_turn |
| CU-069 / P | **Remember a verified file operation**. Setup: Owned Files task has just moved alpha.txt to Destination. Say: “Remember where you moved alpha dot txt under {marker}, then tell me its location on my next request.” | Note derives from current verified move receipt; next spoken recall searches memory and reports the actual new path | pointer, memory, multi_turn |
| CU-070 / P | **Resume a task from a session note**. Setup: Same session with source-bound partial task note and fresh owned files. Say: “Read the checkpoint under {marker}, inspect what is actually here now, and finish the remaining rename and save steps.” | Memory retrieval followed by fresh UI observation and new action IDs; independently verified final files; completed steps not blindly replayed | pointer, keyboard, memory, multi_turn |

## Live Moondream and Kev decisions, uncertainty and recovery

| ID / readiness | Setup and spoken task | Independent success check | Required method |
| --- | --- | --- | --- |
| CU-071 / P | **Ground a positional target**. Setup: Fresh Calculator with observed geometry. Say: “Click the number in the top left of the numeric keypad, then tell me the digit now displayed.” | Pinned Moondream frame-bound point drives actual pointer input; fresh display matches the independently observed button | pointer, vision |
| CU-072 / P | **Ground an icon without a text label**. Setup: Owned Firefox with visible back icon and one prior page. Say: “Click the left-pointing arrow near the top left to return to the previous page.” | Actual Moondream point receipt plus mouse click; independently verified previous committed URL | pointer, vision |
| CU-073 / P | **Re-ground after window movement**. Setup: Owned Calculator; host moves window once after an observation before a pending input. Say: “Click the bottom right Equals button using its appearance, even if the window moves.” | Old geometry/frame action rejects without input; fresh grounding leads to exactly one valid input at new location or explicit unresolved failure | pointer, vision, negative |
| CU-074 / P | **Choose between repeated labels**. Setup: Two owned app panels with same button label but distinct visible context; active Kev. Say: “Click the Apply button in the panel for the scratch document, leaving the other panel alone.” | Actual Kev distribution preserves both contexts; accepted choice maps to correct panel; wrong panel unchanged | pointer, kev |
| CU-075 / P | **Abstain on genuinely ambiguous instruction**. Setup: Owned Files with two visibly similar untargeted documents; active Kev. Say: “Open that document.” | Actual distribution/abstention or clarification; zero dispatch to either document and no invented successful opening | kev, negative |
| CU-076 / P | **Keep low-confidence choices from clicking**. Setup: Real app with ambiguous controls; active Kev confidence left at 0.95. Say: “Choose the control that would save the scratch note, but stop if the current evidence does not identify it confidently.” | Actual low-confidence Kev result causes not_dispatched; no threshold reduction or keyboard bypass; low confidence is not task completion | kev, negative |
| CU-077 / P | **Reject a stale visual candidate**. Setup: Owned Gedit; host changes content once after point offering before execution. Say: “Double click the visible word alpha and verify the selection.” | Old frame-bound candidate rejects without stale click; fresh current observation needed; selected text must match current owned buffer | pointer, vision, negative |
| CU-078 / P | **Handle a covered target**. Setup: Owned Calculator target briefly covered by another owned window. Say: “Click seven in Calculator, bringing Calculator forward first if necessary.” | Occluded hit rejects without input; fresh foreground/hit proof precedes valid click; covering app unchanged | pointer, vision, negative |
| CU-079 / P | **Require a real policy decision in a workflow**. Setup: Owned Calculator with ambiguous finite button offers; active Kev. Say: “Using the offered button choices, calculate six plus nine and verify the answer.” | Live Kev accepted distributions bind each required ambiguous choice to actual dispatch; result 15; semantic/keyboard bypass fails method criterion | pointer, kev |
| CU-080 / P | **Retain uncertain dispatch without replay**. Setup: Owned Gedit; coordinator interrupts one action response after dispatch boundary. Say: “Type copper once in the empty document, and verify what actually appeared before doing anything else.” | Exactly one copper insertion; uncertain action never automatically replayed; fresh independent buffer resolves result or case remains unresolved | keyboard, negative |

## Actual audio, interruptions and call-session behavior

| ID / readiness | Setup and spoken task | Independent success check | Required method |
| --- | --- | --- | --- |
| CU-081 / P | **Recognize a normal conversational request**. Setup: Empty owned Gedit. Say: “Could you put the words copper finch ready into this empty note for me?” | Real Omni TTS WAV and ASR transcript retain task intent; keyboard input creates exact requested buffer; no typed goal substitute | keyboard |
| CU-082 / P | **Recognize an indirect paraphrase**. Setup: Owned Files showing alpha.txt and beta.txt. Say: “I want the first file highlighted, just alpha, and I do not want it opened.” | ASR/comprehension yield single native selection, no open; exact independent selected-node set | pointer |
| CU-083 / P | **Preserve a negated action**. Setup: Owned Gedit with original file hash. Say: “Select all the text, but do not replace it and do not save anything.” | Full selection verified; buffer/file hashes and modification timestamps unchanged; no Save or typing dispatch | keyboard, negative |
| CU-084 / P | **Apply a spoken correction**. Setup: Empty owned Gedit; two generated speech turns within same session. Say: “Turn one: Write copper finch ready. Turn two: Change only finch to robin and leave the other words alone.” | Both real speech inputs and ASR traces exist; final buffer copper robin ready; no stale prior audio or text replay | keyboard, multi_turn |
| CU-085 / P | **Resolve a follow-up referent**. Setup: Owned Files with alpha.txt and Destination; two speech turns. Say: “Turn one: Select alpha dot txt. Turn two: Move that selected file into Destination and verify it.” | Fresh follow-up resolves actual current selected file; source absent, destination original hash; no wrong file moved | pointer, keyboard, multi_turn |
| CU-086 / P | **Ignore actual quiet and click captures**. Setup: Real call input with quiet, click and steady-noise waveforms played through capture path; no spoken task. Say: “No spoken instruction in this case; play the bounded non-speech input and observe the call.” | VAD submits zero model action turns for specified fixtures; no UI dispatch or claimed user transcript; actual capture provenance retained | negative |
| CU-087 / M | **Accept one physical microphone command**. Setup: Physical mic armed through local harness; owned Gedit empty; user speaks after cue. Say: “In the Text Editor, type copper finch ready and leave it visible.” | Actual mic frames, VAD segment, ASR transcript and keyboard input form one trace; exact fresh buffer; never labeled generated TTS | keyboard, physical_microphone |
| CU-088 / M | **Handle acoustic playback through a microphone**. Setup: Omni-generated instruction played over speaker into actual armed mic; owned Calculator. Say: “Use Calculator to work out six plus nine by clicking the buttons and tell me the result.” | Speaker WAV hash plus physical mic recording/VAD/ASR, actual clicks and fresh result 15; acoustic capture distinct from direct WAV submission | pointer, physical_microphone |
| CU-089 / M | **Interrupt ongoing speech with a correction**. Setup: Active spoken reply; user mic or generated speaker correction with real capture; owned Gedit. Say: “Stop that action. Leave the document unchanged and tell me what you have already done.” | Actual barge-in duck/pause and sustained-speech handling; stale inference canceled; no later stale mutation; final cites actual current state | physical_microphone, multi_turn, negative |
| CU-090 / P | **Consolidate several rapid spoken segments**. Setup: Same real call; short generated speech segments captured as one pending request; owned Calculator. Say: “Clear Calculator. Then calculate seventeen plus twenty five. Verify the answer.” | Actual VAD/queue trace stays single-flight with bounded newest-audio consolidation; one coherent task finishes at 42 without overlapping dispatch | pointer, multi_turn |

## Complex tasks across normal desktop applications

| ID / readiness | Setup and spoken task | Independent success check | Required method |
| --- | --- | --- | --- |
| CU-091 / P | **Documentation to saved note**. Setup: Owned Firefox at Python docs, empty Gedit, owned Workbench. Say: “Read a short Numbers fact in Python's tutorial, copy it into the Text Editor with its source address, and save python note dot txt in Workbench.” | Fresh visible source, actual selection/copy/paste or keyboard inputs, exact saved note and actual URL independently read back | pointer, keyboard |
| CU-092 / P | **Calculator to a saved report**. Setup: Owned Calculator, empty Gedit and Workbench. Say: “Calculate seventeen plus twenty five using Calculator, then write the result into a note named total dot txt and save it in Workbench.” | Real Calculator input/result 42 precedes editor entry; saved file has requested result; independent two-app and file receipts | pointer, keyboard |
| CU-093 / P | **Organize and edit a document**. Setup: Owned Files Workbench with alpha.txt and Destination. Say: “Rename alpha dot txt to copper dot txt, open it in the Text Editor, append ready on a new line, save it, and move it into Destination.” | Rename, actual editor mutation/save and GUI move independently verified in order; exact final bytes and path; source absent | pointer, keyboard |
| CU-094 / P | **Read a local PDF and cite its page**. Setup: Owned PDF viewer with two-page public-source test PDF and source URL; empty Gedit. Say: “Open the PDF's second page, select its heading, and save that heading with the page number in a text note.” | Fresh viewer page 2 and actual selected heading; saved note matches source heading and page; no off-screen invented text | pointer, keyboard |
| CU-095 / P | **Make a spreadsheet and export it**. Setup: Owned blank LibreOffice Calc workbook and Workbench. Say: “Enter the three expenses twelve, thirty and eight, total them, save expenses dot ods, and export a PDF in Workbench.” | GUI cells total 50; saved ODS preserves inputs/formula; exported PDF contains expected total; actual export dialogs observed | pointer, keyboard |
| CU-096 / P | **Use retrieved memory in a desktop task**. Setup: Same session has source-verified note under marker; empty Gedit and Workbench. Say: “Recall the documentation fact under {marker}, put it and its source into a new note, and save recalled note dot txt in Workbench.” | Actual memory_search supplies same-session sourced record; actual editor input; exact saved note; no new browser fetch needed | keyboard, memory, multi_turn |
| CU-097 / P | **Compare docs and save a sourced explanation**. Setup: Firefox with Python pathlib/json tabs, empty Gedit and Workbench. Say: “Compare what pathlib and json are for, write a short explanation with both source addresses, and save comparison dot txt.” | Fresh acquisition from each actual tab; saved explanation correctly attributes both modules and URLs; no mixed-source fact | pointer, keyboard |
| CU-098 / P | **Resume after a harmless dialog interrupts**. Setup: Owned multi-app documentation-to-note task; coordinator opens an owned harmless dialog once mid-task. Say: “Finish putting the sourced documentation note into Workbench, closing only the dialog that gets in the way.” | New dialog observed and dismissed once; fresh continuation uses current IDs; one correct final file; no unrelated window closed | pointer, keyboard, negative |
| CU-099 / P | **Verify before cleaning up a multi-app task**. Setup: Owned Firefox, Gedit, Files and Workbench scratch inputs. Say: “Read a Python Numbers fact, save a sourced note, rename it python fact dot txt, then close only the windows opened for this task after verifying the file.” | Complete source acquisition, exact file write/rename, fresh final disk receipt, all owned windows closed and baseline windows untouched | pointer, keyboard |
| CU-100 / P | **Long multi-turn task with remembered checkpoint**. Setup: Same session; owned Firefox, Gedit, Files, Calculator; Workbench; three spoken turns. Say: “Turn one: Read a Python Numbers fact and remember it under {marker}. Turn two: Calculate seventeen plus twenty five and save a note with the fact, its source and the answer. Turn three: Rename the note verified report dot txt, move it into Destination, and check everything.” | Three actual speech/ASR turns, memory read/write, real result 42, source-bound exact saved report and final moved path; no stale input replay; bounded task completion | pointer, keyboard, memory, multi_turn |

## Execution and evidence

Use native `think:false` for the 200-result baseline. Optional `think:true`
comparisons are additional runs, not substitutes for an unsuccessful baseline.
Cases 001–010 reuse `runtime.verify_real_desktop_suite`, whose wrapper starts
neither services nor models. Once the exact selected stack is actually running
on GPU 1, the first bounded case can be invoked as follows:

```bash
.venv/bin/python -m runtime.verify_real_desktop_suite \
  --model '<exact deployed Qwen or Ornith tag>' \
  --endpoint http://127.0.0.1:8920 \
  --token-file '<selected runtime root>/state/access-token.txt' \
  --case gedit-selection-only --timeout 300 --case-timeout 900
```

The endpoint and token path must belong to that exact deployed model; use 9020
only if the alternate validation profile is actually deployed there. Never
put the token itself on the command line or in a committed report. Existing
runner-selected subset results still leave the other 99 plan cases unrun.
Cases 011–100 are specifications, not new executable runner branches.

Keep fresh per-case/model artifacts under ignored `runtime-data/`: TTS WAV or
mic capture; ASR transcript; session-isolated tool trace; exact model/sidecar and
support revisions; lease-scope and process residency evidence; pointer events;
owned-window frames; before/intermediate/after UI states; independently read
file/source/memory receipts; final response; latency and cleanup. Do not commit
private desktop recordings, browser content, credentials or lease tokens.
No historical success is imported into this new matrix.

Useful triage stages are capture/VAD, TTS, ASR, comprehension, capability
discovery, planning, Kev choice, visual grounding, guarded dispatch, effect
verification, terminal oracle, spoken final and cleanup. Preserve the first
failure and classify later unreached stages explicitly. Model self-reported
success and a child exit code are not an oracle.

The public documentation targets were checked while preparing this plan:
[Python introduction](https://docs.python.org/3/tutorial/introduction.html),
[pathlib](https://docs.python.org/3/library/pathlib.html),
[json](https://docs.python.org/3/library/json.html),
[MDN button](https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Elements/button),
[MDN accessibility](https://developer.mozilla.org/en-US/docs/Web/Accessibility),
and [git-status](https://git-scm.com/docs/git-status).
During live cases, reacquire the current page and verify visible content and
actual address; the plan's expected fact cannot substitute for browsing.
