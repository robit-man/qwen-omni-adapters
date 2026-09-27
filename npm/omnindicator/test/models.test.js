"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { evaluateProfiles } = require("../lib/models");

function host(overrides = {}) {
  return {
    platform: "linux",
    tegra: false,
    totalMemoryGiB: 64,
    maxGpuMemoryGiB: 32,
    diskFreeGiB: 100,
    ...overrides,
  };
}

test("small Tegra is rejected before downloads", () => {
  const profiles = evaluateProfiles(host({ tegra: true, totalMemoryGiB: 15, maxGpuMemoryGiB: null }));
  assert.deepEqual(profiles.map((profile) => profile.state), ["unsupported", "unsupported"]);
  assert.match(profiles[0].reason, /29 GiB-class/);
});

test("32 GB-class Tegra admits Ornith and marks Qwen conditional", () => {
  const profiles = evaluateProfiles(host({ tegra: true, totalMemoryGiB: 29.8, maxGpuMemoryGiB: null }));
  assert.equal(profiles[0].state, "supported");
  assert.equal(profiles[1].state, "conditional");
  assert.match(profiles[0].reason, /29 GiB recommendation/);
  assert.match(profiles[1].reason, /48 GiB recommendation/);
});

test("adequate discrete host admits both profiles", () => {
  const profiles = evaluateProfiles(host());
  assert.deepEqual(profiles.map((profile) => profile.state), ["supported", "supported"]);
});

test("discrete host RAM and disk are screened independently", () => {
  const lowRam = evaluateProfiles(host({ totalMemoryGiB: 24 }));
  assert.equal(lowRam[0].state, "supported");
  assert.equal(lowRam[1].state, "unsupported");

  const lowDisk = evaluateProfiles(host({ diskFreeGiB: 20 }));
  assert.match(lowDisk[0].diskWarning, /allow at least 28 GiB/);
  assert.match(lowDisk[1].diskWarning, /allow at least 42 GiB/);
});
