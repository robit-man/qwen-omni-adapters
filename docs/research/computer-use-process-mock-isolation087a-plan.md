# 087a: Isolate the existing cleanup test process mock

The failing criterion is test attribution: the helper-cleanup test must measure
its own child process. Correct isolated full-before89922 ended with5439 passes
and one child-liveness failure. Preserve this actual failed gate.

Three fixed controlled cases pass/fail/pass: original test alone, an imposed
concurrent launch under its global subprocess.Popen patch, and identical
concurrency under a focus-module-local proxy. The failing case killed/reaped
the helper group but overwrote its child-pid file with another group's live
child. That proves a test-isolation defect; the original occurrence lacked
PID-start/group tracing, so its exact cause remains unknown.

Change only the existing fixture in tests/test_passive_focus_guard.py to patch
focus.subprocess with a local namespace and retain the original global Popen,
PIPE and TimeoutExpired. Assert global Popen remains unchanged. Preserve every
150ms/800ms/50x10ms bound, process-group/CUDA and death assertion. No runtime,
cleanup, input, focus, model, schema, lifecycle or policy change is authorized.

Run the existing focused module and an independent concurrent launch control,
freeze the one-file delta, then run the full repository gate with explicit
candidate PYTHONPATH. A green gate is the new before checkpoint for still-held
087 runtime work; the original invalid and failed gates remain literal.
