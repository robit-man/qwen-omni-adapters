"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { parseArgs, renderDoctor, usage } = require("../lib/cli");

test("no arguments means guarded install", () => {
  assert.equal(parseArgs([]).command, "install");
});

test("noninteractive deployment flags are parsed", () => {
  const parsed = parseArgs(["install", "--profile", "ornith15", "--yes", "--core-only", "--dry-run"]);
  assert.equal(parsed.profile, "ornith15");
  assert.equal(parsed.yes, true);
  assert.equal(parsed.coreOnly, true);
  assert.equal(parsed.dryRun, true);
});

test("unknown commands and options fail closed", () => {
  assert.throws(() => parseArgs(["launch"]), /Unknown command/);
  assert.throws(() => parseArgs(["--force"]), /Unknown option/);
});

test("help and doctor disclose platform indicator scope", () => {
  assert.match(usage(), /no native\ntray indicator is claimed/);
  const report = renderDoctor({
    platform: "darwin",
    arch: "arm64",
    tegra: false,
    totalMemoryGiB: 64,
    freeMemoryGiB: 40,
    diskFreeGiB: 100,
    gpus: [],
    brokerManaged: false,
    indicatorAvailable: false,
    desktopSessionDetected: false,
    profiles: [{ id: "ornith15", state: "supported", artifactGiB: 8.15, reason: "fits", diskWarning: null }],
    metalSupported: true,
    supportError: null,
    missingDependencies: [],
    runningAsRoot: false,
  });
  assert.match(report, /not implemented on this platform/);
  assert.match(report, /not a residency guarantee/);
});
