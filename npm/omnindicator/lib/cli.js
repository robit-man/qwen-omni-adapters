"use strict";

const fs = require("node:fs");
const path = require("node:path");
const readline = require("node:readline/promises");
const { stdin, stdout } = require("node:process");
const { inspectHost } = require("./host");
const {
  RELEASE_REF,
  buildDeployment,
  dataRoot,
  ensureRepository,
  runDeployment,
  runStatus,
} = require("./installer");

function usage() {
  return `Usage: omnindicator [install|doctor|status] [options]

With no command, omnindicator runs the guarded interactive installer.

Options:
  --profile ornith15|qwen38  Select a trained audio-bridge model
  --repo-dir PATH            Use an existing checkout without changing its Git state
  --core-only                Linux: omit the always-listening indicator harness
  --yes, -y                  Accept the deployment plan (requires --profile without a TTY)
  --dry-run                  Show the resolved plan without cloning, pulling weights, or starting services
  --json                     Emit machine-readable doctor output
  --help, -h                 Show this help
  --version, -v              Show the package version

The Linux desktop default installs the visible AppIndicator harness. macOS and
Windows currently install the GPU runtime/service and portal only; no native
tray indicator is claimed on those platforms.`;
}

function parseArgs(argv) {
  const result = {
    command: "install",
    profile: null,
    repoDir: null,
    externalCheckout: false,
    coreOnly: false,
    yes: false,
    dryRun: false,
    json: false,
    help: false,
    version: false,
  };
  let commandSeen = false;
  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index];
    if (!value.startsWith("-") && !commandSeen) {
      if (!new Set(["install", "doctor", "status"]).has(value)) throw new Error(`Unknown command: ${value}`);
      result.command = value;
      commandSeen = true;
    } else if (value === "--profile") {
      result.profile = argv[++index];
      if (!result.profile) throw new Error("--profile requires a value");
    } else if (value === "--repo-dir") {
      result.repoDir = path.resolve(argv[++index] || "");
      if (!argv[index]) throw new Error("--repo-dir requires a value");
      result.externalCheckout = true;
    } else if (value === "--core-only") result.coreOnly = true;
    else if (value === "--yes" || value === "-y") result.yes = true;
    else if (value === "--dry-run") result.dryRun = true;
    else if (value === "--json") result.json = true;
    else if (value === "--help" || value === "-h") result.help = true;
    else if (value === "--version" || value === "-v") result.version = true;
    else throw new Error(`Unknown option: ${value}`);
  }
  return result;
}

function packageVersion() {
  const packagePath = path.resolve(__dirname, "..", "..", "..", "package.json");
  return JSON.parse(fs.readFileSync(packagePath, "utf8")).version;
}

function round1(value) {
  return Number.isFinite(value) ? (Math.round(value * 10) / 10).toFixed(1) : "unknown";
}

function platformLabel(host) {
  if (host.tegra) return `Linux/${host.arch} NVIDIA ${host.tegraSoc || "Tegra"} (unified memory)`;
  if (host.platform === "darwin") return `macOS/${host.arch} Metal`;
  if (host.platform === "win32") return `Windows/${host.arch} NVIDIA CUDA`;
  return `Linux/${host.arch} ${host.brokerManaged ? "broker-managed NVIDIA" : "direct NVIDIA"}`;
}

function renderDoctor(host) {
  const lines = [
    `Host: ${platformLabel(host)}`,
    `System memory: ${round1(host.totalMemoryGiB)} GiB total, ${round1(host.freeMemoryGiB)} GiB currently free`,
    `Install filesystem: ${round1(host.diskFreeGiB)} GiB free`,
  ];
  if (host.gpus.length) {
    for (const gpu of host.gpus) lines.push(`GPU ${gpu.index}: ${gpu.name}, ${round1(gpu.memoryGiB)} GiB`);
  }
  lines.push(`Runtime mode: ${host.tegra ? "direct Tegra" : host.brokerManaged ? "ollama-unify broker" : "platform-native direct"}`);
  lines.push(`Desktop indicator: ${host.indicatorAvailable ? "Linux harness available; target desktop is proven during deployment" : "not implemented on this platform (core runtime/portal only)"}`);
  if (host.platform === "linux") lines.push(`Desktop session signal: ${host.desktopSessionDetected ? "detected" : "not visible from this shell; native installer will attempt signed-in-session recovery"}`);
  lines.push("");
  lines.push("Model screening (coarse pre-download check):");
  for (const profile of host.profiles) {
    lines.push(`  ${profile.id.padEnd(9)} ${profile.state.padEnd(11)} ${profile.artifactGiB} GiB artifacts — ${profile.reason}`);
    if (profile.diskWarning) lines.push(`              disk warning: ${profile.diskWarning}`);
  }
  if (host.supportError) lines.push(`\nBLOCKED: ${host.supportError}`);
  if (host.missingDependencies.length) lines.push(`\nBLOCKED: missing host tools: ${host.missingDependencies.join(", ")}`);
  if (host.runningAsRoot && host.platform === "linux") lines.push("\nWARNING: run omnindicator as the signed-in desktop user, not root; deploy.sh requests sudo only for system steps.");
  lines.push("\nThis is an early screen, not a residency guarantee. The repository doctor, exact layer-byte admission, live memory reserve, GPU-residency proof, and service readiness gates still run on the target host.");
  return lines.join("\n");
}

function doctorJson(host) {
  return {
    platform: host.platform,
    arch: host.arch,
    tegra: host.tegra,
    tegraSoc: host.tegraSoc,
    totalMemoryGiB: host.totalMemoryGiB,
    freeMemoryGiB: host.freeMemoryGiB,
    diskFreeGiB: host.diskFreeGiB,
    gpus: host.gpus.map(({ index, name, memoryGiB, driver }) => ({ index, name, memoryGiB, driver })),
    maxGpuMemoryGiB: host.maxGpuMemoryGiB,
    metalSupported: host.metalSupported,
    brokerManaged: host.brokerManaged,
    indicatorAvailable: host.indicatorAvailable,
    desktopSessionDetected: host.desktopSessionDetected,
    dependencies: Object.fromEntries(Object.entries(host.dependencies).map(([name, value]) => [name, Boolean(value)])),
    missingDependencies: host.missingDependencies,
    supportError: host.supportError,
    runningAsRoot: host.runningAsRoot,
    profiles: host.profiles,
  };
}

async function ask(prompt) {
  const interface_ = readline.createInterface({ input: stdin, output: stdout });
  try {
    return (await interface_.question(prompt)).trim();
  } finally {
    interface_.close();
  }
}

async function chooseProfile(host, requested) {
  if (requested) {
    const fit = host.profiles.find((profile) => profile.id === requested);
    if (!fit) throw new Error(`Unknown profile ${requested}; expected ornith15 or qwen38.`);
    return fit;
  }
  if (!stdin.isTTY || !stdout.isTTY) throw new Error("Non-interactive installation requires --profile and --yes.");
  const candidates = host.profiles.filter((profile) => profile.state !== "unsupported");
  if (!candidates.length) throw new Error("This host did not pass the coarse compute floor for either model.");
  stdout.write("\nChoose a model:\n");
  candidates.forEach((profile, index) => stdout.write(`  ${index + 1}) ${profile.title} [${profile.state}]\n`));
  const answer = await ask(`Selection [1]: `);
  const choice = answer === "" ? 0 : Number(answer) - 1;
  if (!Number.isInteger(choice) || choice < 0 || choice >= candidates.length) throw new Error("Invalid model selection.");
  return candidates[choice];
}

function shellDisplay(command, args) {
  return [command, ...args].map((value) => /[^A-Za-z0-9_./:=+-]/.test(value) ? JSON.stringify(value) : value).join(" ");
}

async function install(options, host) {
  if (host.supportError) throw new Error(host.supportError);
  if (host.missingDependencies.length) {
    throw new Error(`Install prerequisites are missing: ${host.missingDependencies.join(", ")}. No repository or model data was downloaded.`);
  }
  if (host.platform === "linux" && host.runningAsRoot && !options.coreOnly) {
    throw new Error("Refusing a root-owned desktop indicator. Run as the signed-in desktop user; deploy.sh will request sudo when needed.");
  }
  if (host.platform === "linux" && !options.coreOnly && !host.desktopSessionDetected) {
    throw new Error("No graphical desktop session was detected for the indicator. Run from the signed-in desktop session or pass --core-only; no repository or model data was downloaded.");
  }

  const profile = await chooseProfile(host, options.profile);
  if (profile.state === "unsupported") throw new Error(`${profile.title} is not admitted: ${profile.reason}`);
  if (profile.diskWarning) throw new Error(`${profile.diskWarning} No repository or model data was downloaded.`);

  let coreOnly = options.coreOnly;
  if (!host.indicatorAvailable && !coreOnly) {
    stdout.write("\nThis platform has no native omnindicator tray implementation. The supported deployment is the core GPU service and portal.\n");
    if (!options.yes) {
      if (!stdin.isTTY || !stdout.isTTY) throw new Error("Pass --core-only --yes to acknowledge core-only deployment on this platform.");
      const answer = (await ask("Continue with the core runtime and portal only? [y/N] ")).toLowerCase();
      if (answer !== "y" && answer !== "yes") return;
    }
    coreOnly = true;
  }

  const root = dataRoot(host.platform);
  const repoDir = options.repoDir || path.join(root, "qwen-omni-adapters");
  const planSummary = [
    "\nDeployment plan",
    `  Release:    omnindicator ${packageVersion()} / ${RELEASE_REF}`,
    `  Checkout:   ${repoDir}`,
    `  Model:      ${profile.title}`,
    `  Artifacts:  ${profile.artifactGiB} GiB plus KV/workspaces/runtime reserve`,
    `  Interface:  ${coreOnly ? "core service + portal" : "core service + always-listening Linux indicator"}`,
    `  Screening:  ${profile.state} — ${profile.reason}`,
  ].join("\n");
  stdout.write(`${planSummary}\n`);

  if (!options.yes) {
    if (!stdin.isTTY || !stdout.isTTY) throw new Error("Non-interactive installation requires --yes.");
    const answer = (await ask("Proceed with checkout and native deployment? [y/N] ")).toLowerCase();
    if (answer !== "y" && answer !== "yes") return;
  }

  const existing = fs.existsSync(repoDir);
  const preview = buildDeployment({ platform: host.platform, repoDir, profile: profile.id, coreOnly, existing });
  if (options.dryRun) {
    stdout.write(`\nDry run: would prepare ${options.externalCheckout ? "the supplied checkout" : `the release-pinned ${RELEASE_REF} checkout`} and run:\n  ${shellDisplay(preview.command, preview.args)}\n`);
    return;
  }

  const checkout = ensureRepository({
    repoDir,
    ref: RELEASE_REF,
    externalCheckout: options.externalCheckout,
    git: host.dependencies.git,
  });
  const deployment = buildDeployment({
    platform: host.platform,
    repoDir: checkout.repoDir,
    profile: profile.id,
    coreOnly,
    existing: !checkout.created,
  });
  stdout.write(`\nStarting the repository-native deployment. Pulls begin only after the coarse host gate above; the native deployer then validates artifacts, live memory, GPU residency, and readiness before cutover.\n`);
  runDeployment(deployment);
  stdout.write(`\nomnindicator deployment completed. Run \"omnindicator status\" for current component state.\n`);
}

async function main(argv) {
  const options = parseArgs(argv);
  if (options.help) {
    stdout.write(`${usage()}\n`);
    return;
  }
  if (options.version) {
    stdout.write(`${packageVersion()}\n`);
    return;
  }
  if (options.command === "status") {
    const repoDir = options.repoDir || path.join(dataRoot(process.platform), "qwen-omni-adapters");
    runStatus(repoDir, process.platform);
    return;
  }

  const host = inspectHost({ installTarget: options.repoDir ? path.dirname(options.repoDir) : dataRoot(process.platform) });
  if (options.command === "doctor") {
    stdout.write(options.json ? `${JSON.stringify(doctorJson(host), null, 2)}\n` : `${renderDoctor(host)}\n`);
    return;
  }

  stdout.write("omnindicator — guarded installer for the Qwen Omni local voice, perception, tool, and desktop-indicator runtime\n\n");
  stdout.write(`${renderDoctor(host)}\n`);
  await install(options, host);
}

module.exports = { chooseProfile, doctorJson, install, main, packageVersion, parseArgs, renderDoctor, usage };
