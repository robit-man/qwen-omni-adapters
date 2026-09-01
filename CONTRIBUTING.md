# Contributing

Keep changes narrow and preserve the invariants in `AGENTS.md`. Run:

```bash
./scripts/bootstrap.sh --skip-llama --skip-models
./scripts/validate.sh
```

Protocol, schema, parser, clients, runtime, portal, and tests are one
compatibility surface. Update them together when a public field or event
changes. Never commit weights, component caches, access tokens, session logs,
or generated tunnel URLs.
