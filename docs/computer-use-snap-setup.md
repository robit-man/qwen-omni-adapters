# Installed Snap Chromium setup

The long-horizon runner accepts `--browser chromium-snap` as an explicit host
condition. It selects the existing `/snap/bin/chromium` alias, validates its
root-owned executable ELF dispatcher at `/usr/bin/snap`, and launches only
`--new-window about:blank`. It preserves the ordinary browser profile, PATH,
MIME defaults and launch flags. There is no installation or update step.

Selection delegates to the existing ordinary-app registration, fresh native
observation, task scope and exact polite-cleanup guards. A valid launcher does
not prove browser ownership or accessibility. Missing, changed or invalid
launcher identity, permissions or content metadata blocks before launch. Access-time
changes from reading the installed file are excluded. An unproved window remains unregistered
and cannot be closed on guessed identity. Flatpak setup and its earlier failure
receipts are separate.

CPU tests verify this declared selection and unchanged admission/cleanup paths.
Actual Snap browser/provider ownership and a whole-task result remain untested
until a separately admitted canonical run. The adapter/model tool contract and
packaged Goal, inputs, criteria and execution budgets are unchanged.
