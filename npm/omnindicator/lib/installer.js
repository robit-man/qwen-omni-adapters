"use strict";

const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { spawnSync } = require("node:child_process");
const { getModel } = require("./models");

const REPOSITORY_URL = "https://github.com/robit-man/qwen-omni-adapters.git";
const RELEASE_REF = "npm-v0.1.3";

function dataRoot(platform = process.platform, env = process.env) {
  if (env.OMNINDICATOR_HOME) return path.resolve(env.OMNINDICATOR_HOME);
  if (platform === "win32") {
    return path.join(env.LOCALAPPDATA || path.join(os.homedir(), "AppData", "Local"), "omnindicator");
  }
  if (platform === "darwin") {
    return path.join(os.homedir(), "Library", "Application Support", "omnindicator");
  }
  return path.join(env.XDG_DATA_HOME || path.join(os.homedir(), ".local", "share"), "omnindicator");
}

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: options.cwd,
    env: options.env || process.env,
    encoding: "utf8",
    stdio: options.inherit ? "inherit" : "pipe",
    windowsHide: true,
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    const detail = options.inherit ? "" : `: ${(result.stderr || result.stdout || "").trim()}`;
    throw new Error(`${path.basename(command)} ${args.join(" ")} failed${detail}`);
  }
  return (result.stdout || "").trim();
}

function normalizeRemote(value) {
  return value
    .trim()
    .replace(/^git@github\.com:/, "https://github.com/")
    .replace(/^ssh:\/\/git@github\.com\//, "https://github.com/")
    .replace(/\.git$/, "")
    .replace(/\/$/, "")
    .toLowerCase();
}

function validateCheckout(repoDir) {
  const expected = ["AGENTS.md", "deploy.sh", "deploy-macos.sh", "deploy.ps1"];
  const missing = expected.filter((name) => !fs.existsSync(path.join(repoDir, name)));
  if (missing.length) throw new Error(`${repoDir} is not a complete qwen-omni-adapters checkout (missing ${missing.join(", ")}).`);
}

function validateOrigin(repoDir, git) {
  const remote = run(git, ["remote", "get-url", "origin"], { cwd: repoDir });
  if (normalizeRemote(remote) !== normalizeRemote(REPOSITORY_URL)) {
    throw new Error(`Refusing to execute an unexpected checkout origin: ${remote}.`);
  }
}

function releaseTarget(repoDir, git, ref) {
  run(git, ["fetch", "--force", "--depth=100", "origin", "main", `refs/tags/${ref}:refs/tags/${ref}`], { cwd: repoDir });
  const target = run(git, ["rev-list", "-n", "1", ref], { cwd: repoDir });
  const belongsToMain = spawnSync(git, ["merge-base", "--is-ancestor", target, "origin/main"], { cwd: repoDir }).status === 0;
  if (!belongsToMain) throw new Error(`Release ${ref} is not reachable from origin/main; refusing the checkout.`);
  return target;
}

function ensureRepository(options) {
  const repoDir = path.resolve(options.repoDir);
  const root = path.dirname(repoDir);
  const statePath = path.join(root, "install-state.json");
  const ref = options.ref || RELEASE_REF;
  const git = options.git || "git";

  if (fs.existsSync(repoDir)) {
    validateCheckout(repoDir);
    validateOrigin(repoDir, git);
    if (options.externalCheckout) return { repoDir, created: false, updated: false, source: "existing" };

    if (!fs.existsSync(statePath)) {
      throw new Error(`${repoDir} already exists but is not marked as an omnindicator-managed checkout; pass --repo-dir explicitly to use it without changing Git state.`);
    }
    const dirty = run(git, ["status", "--porcelain", "--untracked-files=no"], { cwd: repoDir });
    if (dirty) throw new Error("Tracked checkout changes are present; refusing to move the managed release checkout.");
    const branch = run(git, ["branch", "--show-current"], { cwd: repoDir });
    if (branch !== "main") throw new Error(`Managed checkout must remain on main for indicator updates (found ${branch || "detached HEAD"}).`);
    const target = releaseTarget(repoDir, git, ref);
    const head = run(git, ["rev-parse", "HEAD"], { cwd: repoDir });
    let updated = false;
    if (head !== target) {
      const targetIsOlder = spawnSync(git, ["merge-base", "--is-ancestor", target, head], { cwd: repoDir }).status === 0;
      if (!targetIsOlder) {
        run(git, ["merge", "--ff-only", target], { cwd: repoDir });
        updated = true;
      }
    }
    return { repoDir, created: false, updated, source: ref };
  }

  fs.mkdirSync(root, { recursive: true, mode: 0o700 });
  run(git, ["clone", "--branch", "main", "--single-branch", "--filter=blob:none", REPOSITORY_URL, repoDir], { inherit: true });
  const target = releaseTarget(repoDir, git, ref);
  run(git, ["checkout", "-B", "main", target], { cwd: repoDir });
  run(git, ["branch", "--set-upstream-to=origin/main", "main"], { cwd: repoDir });
  validateCheckout(repoDir);
  fs.writeFileSync(statePath, `${JSON.stringify({ repository: REPOSITORY_URL, ref, repoDir }, null, 2)}\n`, { mode: 0o600 });
  return { repoDir, created: true, updated: false, source: ref };
}

function buildDeployment(options) {
  const model = getModel(options.profile);
  if (!model) throw new Error(`Unknown model profile: ${options.profile}`);
  const environment = {
    ...process.env,
    OMNI_PROFILE: model.id,
    OMNI_MODEL: model.tag,
    OMNI_LANGUAGE_MODEL: model.tag,
  };

  if (options.platform === "linux") {
    return {
      command: "bash",
      args: [
        path.join(options.repoDir, "deploy.sh"),
        "--profile", model.id,
        "--action", options.existing ? "deploy" : "install",
        options.coreOnly ? "--no-harness" : "--with-harness",
        "--yes",
      ],
      cwd: options.repoDir,
      env: environment,
      indicator: !options.coreOnly,
    };
  }
  if (options.platform === "darwin") {
    return {
      command: "bash",
      args: [path.join(options.repoDir, "deploy-macos.sh")],
      cwd: options.repoDir,
      env: environment,
      indicator: false,
    };
  }
  return {
    command: "powershell",
    args: ["-NoProfile", "-ExecutionPolicy", "Bypass", "-File", path.join(options.repoDir, "deploy.ps1")],
    cwd: options.repoDir,
    env: environment,
    indicator: false,
  };
}

function runDeployment(plan) {
  run(plan.command, plan.args, { cwd: plan.cwd, env: plan.env, inherit: true });
}

function runStatus(repoDir, platform) {
  const executable = platform === "win32"
    ? path.join(repoDir, ".venv", "Scripts", "qwen-omni-daemon.exe")
    : path.join(repoDir, ".venv", "bin", "qwen-omni-daemon");
  if (!fs.existsSync(executable)) throw new Error(`No deployed runtime found at ${repoDir}.`);
  run(executable, ["status"], { cwd: repoDir, inherit: true });
}

module.exports = {
  RELEASE_REF,
  REPOSITORY_URL,
  buildDeployment,
  dataRoot,
  ensureRepository,
  normalizeRemote,
  runDeployment,
  runStatus,
  validateCheckout,
  validateOrigin,
};
