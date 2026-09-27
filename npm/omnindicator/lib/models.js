"use strict";

const MODELS = Object.freeze([
  Object.freeze({
    id: "ornith15",
    title: "Standard Ornith 1.5 9B audio bridge",
    tag: "robit/ornith-1.5-omni-audio-bridge:q4km",
    artifactGiB: 8.15,
    diskFloorGiB: 28,
    hostMemoryFloorGiB: 16,
    discreteFloorGiB: 16,
    discreteRecommendedGiB: 20,
    tegraFloorGiB: 29,
    tegraRecommendedGiB: 29,
    macFloorGiB: 29,
    macRecommendedGiB: 32,
  }),
  Object.freeze({
    id: "qwen38",
    title: "Qwen3.8 27B E03 Obliterated audio bridge",
    tag: "robit/qwen3.8-27b-e03-obliterated-omni-audio-bridge:q4km",
    artifactGiB: 18.33,
    diskFloorGiB: 42,
    hostMemoryFloorGiB: 32,
    discreteFloorGiB: 24,
    discreteRecommendedGiB: 32,
    tegraFloorGiB: 29,
    tegraRecommendedGiB: 48,
    macFloorGiB: 32,
    macRecommendedGiB: 48,
  }),
]);

function round1(value) {
  return Math.round(value * 10) / 10;
}

function profileFit(model, host) {
  const diskWarning = Number.isFinite(host.diskFreeGiB) && host.diskFreeGiB < model.diskFloorGiB
    ? `Only ${round1(host.diskFreeGiB)} GiB is free on the install filesystem; allow at least ${model.diskFloorGiB} GiB.`
    : null;

  let available;
  let floor;
  let recommended;
  let resource;

  if (!host.tegra && host.platform !== "darwin" && host.totalMemoryGiB < model.hostMemoryFloorGiB) {
    return {
      ...model,
      state: "unsupported",
      reason: `${round1(host.totalMemoryGiB)} GiB system memory detected; this profile requires at least ${model.hostMemoryFloorGiB} GiB of host memory in addition to GPU capacity.`,
      diskWarning,
    };
  }

  if (host.tegra) {
    available = host.totalMemoryGiB;
    floor = model.tegraFloorGiB;
    recommended = model.tegraRecommendedGiB;
    resource = "unified memory";
  } else if (host.platform === "darwin") {
    available = host.totalMemoryGiB;
    floor = model.macFloorGiB;
    recommended = model.macRecommendedGiB;
    resource = "unified memory";
  } else {
    available = host.maxGpuMemoryGiB;
    floor = model.discreteFloorGiB;
    recommended = model.discreteRecommendedGiB;
    resource = "GPU memory";
  }

  if (!Number.isFinite(available) || available <= 0) {
    return {
      ...model,
      state: "unsupported",
      reason: `No usable ${resource} capacity was detected.`,
      diskWarning,
    };
  }
  if (available < floor) {
    return {
      ...model,
      state: "unsupported",
      reason: `${round1(available)} GiB ${resource} detected; this profile requires at least a ${floor} GiB-class device.`,
      diskWarning,
    };
  }
  if (available < recommended) {
    return {
      ...model,
      state: "conditional",
      reason: `${round1(available)} GiB ${resource} passes the coarse floor but is below the ${recommended} GiB recommendation; the native live-admission gate remains authoritative.`,
      diskWarning,
    };
  }
  return {
    ...model,
    state: "supported",
    reason: `${round1(available)} GiB ${resource} meets the ${recommended} GiB recommendation; live load and residency checks still run before service cutover.`,
    diskWarning,
  };
}

function evaluateProfiles(host) {
  return MODELS.map((model) => profileFit(model, host));
}

function getModel(id) {
  return MODELS.find((model) => model.id === id) || null;
}

module.exports = { MODELS, evaluateProfiles, getModel, profileFit };
