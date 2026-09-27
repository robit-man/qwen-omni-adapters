"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");
const { RELEASE_REF, buildDeployment, dataRoot, normalizeRemote } = require("../lib/installer");

test("Linux plan enables the indicator unless core-only is explicit", () => {
  const plan = buildDeployment({
    platform: "linux",
    repoDir: "/opt/omnindicator/qwen-omni-adapters",
    profile: "ornith15",
    coreOnly: false,
    existing: false,
  });
  assert.equal(plan.indicator, true);
  assert.ok(plan.args.includes("--with-harness"));
  assert.ok(plan.args.includes("install"));
  assert.equal(plan.env.OMNI_MODEL, "robit/ornith-1.5-omni-audio-bridge:q4km");
  assert.equal(plan.env.OMNI_LANGUAGE_MODEL, plan.env.OMNI_MODEL);
});

test("Linux core-only is always an explicit plan flag", () => {
  const plan = buildDeployment({
    platform: "linux",
    repoDir: "/opt/omnindicator/qwen-omni-adapters",
    profile: "ornith15",
    coreOnly: true,
    existing: true,
  });
  assert.equal(plan.indicator, false);
  assert.ok(plan.args.includes("--no-harness"));
  assert.ok(plan.args.includes("deploy"));
});

test("macOS plan is honest about core-only service deployment", () => {
  const plan = buildDeployment({
    platform: "darwin",
    repoDir: "/tmp/qwen-omni-adapters",
    profile: "qwen38",
    coreOnly: true,
    existing: false,
  });
  assert.equal(plan.indicator, false);
  assert.match(plan.args[0], /deploy-macos\.sh$/);
});

test("repository normalization accepts canonical GitHub transports", () => {
  assert.equal(normalizeRemote("git@github.com:robit-man/qwen-omni-adapters.git"), normalizeRemote("https://github.com/robit-man/qwen-omni-adapters.git"));
});

test("custom data root remains explicit", () => {
  assert.equal(dataRoot("linux", { OMNINDICATOR_HOME: "/srv/omni" }), path.resolve("/srv/omni"));
});

test("npm version and source release tag stay synchronized", () => {
  const packageJson = require("../../../package.json");
  assert.equal(RELEASE_REF, `npm-v${packageJson.version}`);
});
