"use strict";

const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { spawnSync } = require("node:child_process");
const { evaluateProfiles } = require("./models");

const GIB = 1024 ** 3;

function executableCandidates(name, platform = process.platform) {
  if (platform !== "win32") return [name];
  const extensions = (process.env.PATHEXT || ".EXE;.CMD;.BAT;.COM")
    .split(";")
    .filter(Boolean);
  return path.extname(name) ? [name] : extensions.map((extension) => `${name}${extension.toLowerCase()}`);
}

function findCommand(name, env = process.env, platform = process.platform) {
  const searchPath = env.PATH || env.Path || env.path || "";
  for (const directory of searchPath.split(path.delimiter).filter(Boolean)) {
    for (const candidate of executableCandidates(name, platform)) {
      const target = path.join(directory, candidate);
      try {
        fs.accessSync(target, fs.constants.X_OK);
        return target;
      } catch (_) {
        // Keep searching.
      }
    }
  }
  return null;
}

function readTegraCompatible() {
  try {
    return fs.readFileSync("/proc/device-tree/compatible").toString("utf8").replaceAll("\0", "\n");
  } catch (_) {
    return "";
  }
}

function detectNvidiaGpus(command) {
  if (!command) return [];
  const result = spawnSync(command, [
    "--query-gpu=index,name,memory.total,driver_version",
    "--format=csv,noheader,nounits",
  ], { encoding: "utf8", timeout: 8000, windowsHide: true });
  if (result.status !== 0) return [];
  return result.stdout.split(/\r?\n/).filter(Boolean).map((line) => {
    const [index, name, memoryMiB, driver] = line.split(",").map((value) => value.trim());
    return {
      index: Number(index),
      name,
      memoryGiB: Number(memoryMiB) / 1024,
      driver,
    };
  }).filter((gpu) => Number.isFinite(gpu.memoryGiB));
}

function detectBroker(dockerCommand) {
  if (!dockerCommand || process.platform !== "linux") return false;
  const result = spawnSync(dockerCommand, ["gpu", "discover"], {
    encoding: "utf8",
    timeout: 10000,
    windowsHide: true,
  });
  return result.status === 0;
}

function detectMacMetal(systemProfilerCommand, platform = process.platform) {
  if (!systemProfilerCommand || platform !== "darwin") return null;
  const result = spawnSync(systemProfilerCommand, ["SPDisplaysDataType"], {
    encoding: "utf8",
    timeout: 10000,
    windowsHide: true,
  });
  if (result.status !== 0) return false;
  return /Metal Support:\s*(?:Supported|Metal\b)/i.test(result.stdout);
}

function diskFreeGiB(target) {
  let candidate = path.resolve(target);
  while (!fs.existsSync(candidate)) {
    const parent = path.dirname(candidate);
    if (parent === candidate) return null;
    candidate = parent;
  }
  try {
    const stats = fs.statfsSync(candidate);
    const available = Number(stats.bavail ?? stats.bfree);
    return available * Number(stats.bsize) / GIB;
  } catch (_) {
    return null;
  }
}

function hasLinuxDesktopSession(env = process.env) {
  if (env.DISPLAY || env.WAYLAND_DISPLAY) return true;
  const uid = typeof process.getuid === "function" ? process.getuid() : null;
  if (uid === null) return false;
  const runtime = env.XDG_RUNTIME_DIR || `/run/user/${uid}`;
  try {
    const entries = fs.readdirSync(runtime);
    const displaySocket = entries.some((entry) => entry.startsWith("wayland-"));
    const bus = fs.existsSync(path.join(runtime, "bus"));
    if (displaySocket && bus) return true;
  } catch (_) {
    // No recoverable user runtime directory.
  }
  try {
    return fs.readdirSync("/tmp/.X11-unix").some((entry) => entry.startsWith("X"));
  } catch (_) {
    return false;
  }
}

function requiredCommands(platform, brokerManaged) {
  if (platform === "win32") {
    return ["git", "cmake", "ffmpeg", "ollama", "powershell", "py"];
  }
  if (platform === "darwin") {
    return ["bash", "git", "cmake", "ffmpeg", "ollama", "python3", "system_profiler", "xcrun"];
  }
  const commands = ["bash", "git", "cmake", "ffmpeg", "ollama", "python3", "sudo", "systemctl"];
  if (brokerManaged) commands.push("docker", "jq", "ss");
  return commands;
}

function platformSupport(platform, arch, tegra) {
  if (!new Set(["linux", "darwin", "win32"]).has(platform)) {
    return `Unsupported operating system: ${platform}.`;
  }
  if (tegra && arch !== "arm64") {
    return `Tegra was detected with unexpected architecture ${arch}.`;
  }
  if (platform === "darwin" && !new Set(["arm64", "x64"]).has(arch)) {
    return `Unsupported macOS architecture: ${arch}.`;
  }
  if (platform === "win32" && arch !== "x64") {
    return `Windows CUDA deployment currently requires x64, not ${arch}.`;
  }
  return null;
}

function inspectHost(options = {}) {
  const platform = options.platform || process.platform;
  const arch = options.arch || process.arch;
  const env = options.env || process.env;
  const compatible = platform === "linux" ? readTegraCompatible() : "";
  const tegra = /^nvidia,tegra\d+/m.test(compatible);
  const nvidiaSmi = findCommand("nvidia-smi", env, platform);
  const gpus = tegra ? [] : detectNvidiaGpus(nvidiaSmi);
  const systemProfiler = findCommand("system_profiler", env, platform);
  const metalSupported = platform === "darwin" ? detectMacMetal(systemProfiler, platform) : null;
  const docker = findCommand("docker", env, platform);
  const brokerManaged = !tegra && detectBroker(docker);
  const totalMemoryGiB = os.totalmem() / GIB;
  const freeMemoryGiB = os.freemem() / GIB;
  const maxGpuMemoryGiB = gpus.length ? Math.max(...gpus.map((gpu) => gpu.memoryGiB)) : null;
  const dependencyNames = requiredCommands(platform, brokerManaged);
  const dependencies = Object.fromEntries(dependencyNames.map((name) => [name, findCommand(name, env, platform)]));
  const missingDependencies = Object.entries(dependencies).filter(([, value]) => !value).map(([name]) => name);
  let supportError = platformSupport(platform, arch, tegra);
  if (!supportError && platform === "darwin" && metalSupported !== true) {
    supportError = "A Metal-capable macOS GPU was not detected.";
  }
  if (!supportError && !tegra && platform !== "darwin" && gpus.length === 0) {
    supportError = "No discrete NVIDIA CUDA GPU was detected.";
  }
  const indicatorAvailable = platform === "linux";
  const desktopSessionDetected = platform === "linux" && hasLinuxDesktopSession(env);
  const installTarget = options.installTarget || os.homedir();

  const host = {
    platform,
    arch,
    tegra,
    tegraSoc: (compatible.match(/^nvidia,(tegra\d+)/m) || [])[1] || null,
    totalMemoryGiB,
    freeMemoryGiB,
    diskFreeGiB: diskFreeGiB(installTarget),
    gpus,
    maxGpuMemoryGiB,
    metalSupported,
    brokerManaged,
    indicatorAvailable,
    desktopSessionDetected,
    dependencies,
    missingDependencies,
    supportError,
    runningAsRoot: platform !== "win32" && typeof process.getuid === "function" && process.getuid() === 0,
  };
  host.profiles = evaluateProfiles(host);
  return host;
}

module.exports = {
  detectNvidiaGpus,
  detectMacMetal,
  diskFreeGiB,
  findCommand,
  hasLinuxDesktopSession,
  inspectHost,
  platformSupport,
  requiredCommands,
};
