#!/usr/bin/env bash
# Redeploy this checkout whenever origin/main moves.
#
# Run by the omni-auto-update user timer that deploy.sh installs. A checkout
# installed from a GitHub tarball has no .git; the first update converts it
# into a real checkout in place (venvs, models and runtime-data are untracked
# and stay), after which every update is a fast-forward. Local edits to
# tracked files or a diverged branch stop the update instead of being
# overwritten. The redeploy reuses the persisted profile and harness choice.
set -Eeuo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
REPO_URL=${OMNI_UPDATE_REPO:-https://github.com/robit-man/qwen-omni-adapters.git}
BRANCH=${OMNI_UPDATE_BRANCH:-main}
DEPLOY=${OMNI_UPDATE_DEPLOY:-$REPO_ROOT/deploy.sh}
LOCK=${XDG_RUNTIME_DIR:-/tmp}/omni-auto-update.lock

log() { printf 'auto-update: %s\n' "$*"; }

exec 9>"$LOCK"
flock -n 9 || { log 'another update is running'; exit 0; }

remote=$(git ls-remote "$REPO_URL" "refs/heads/$BRANCH" | cut -f1)
[[ $remote =~ ^[0-9a-f]{40}$ ]] || { log "could not read $BRANCH from $REPO_URL"; exit 1; }

# .deployed-commit is written only after a successful deploy, so a failed
# redeploy is retried on the next run even though HEAD already moved.
current=$(cat "$REPO_ROOT/.deployed-commit" 2>/dev/null || true)
if [[ $current == "$remote" ]]; then
  exit 0
fi
log "origin/$BRANCH moved to ${remote:0:12} (deployed: ${current:-none})"

if [[ ${OMNI_UPDATE_SKIP_SUDO_CHECK:-0} != 1 ]] && ! sudo -n true 2>/dev/null; then
  log 'unattended redeploy needs passwordless sudo; rerun ./deploy.sh interactively to enable it'
  exit 1
fi

if [[ ! -d $REPO_ROOT/.git ]]; then
  log 'converting the tarball install into a git checkout'
  git -C "$REPO_ROOT" init -q -b "$BRANCH"
  git -C "$REPO_ROOT" remote add origin "$REPO_URL"
  git -C "$REPO_ROOT" fetch -q origin "$BRANCH"
  # Tracked files are replaced by the published revision; untracked and
  # ignored files (venvs, runtime-data, built llama.cpp, models) are kept.
  git -C "$REPO_ROOT" reset -q --hard "origin/$BRANCH"
  git -C "$REPO_ROOT" branch -q --set-upstream-to "origin/$BRANCH"
else
  if [[ -n $(git -C "$REPO_ROOT" status --porcelain --untracked-files=no) ]]; then
    log 'tracked files have local changes; not updating'
    exit 1
  fi
  git -C "$REPO_ROOT" fetch -q origin "$BRANCH"
  git -C "$REPO_ROOT" merge -q --ff-only "origin/$BRANCH" \
    || { log "local branch has diverged from origin/$BRANCH; not updating"; exit 1; }
fi

profile=$(sed -n 's/^OMNI_PROFILE=//p' "$REPO_ROOT/.env" 2>/dev/null | tail -n1)
profile=${profile//\"/}
[[ -n $profile ]] || { log 'no deployed profile recorded in .env; run ./deploy.sh once'; exit 1; }
harness=--no-harness
if systemctl --user is-enabled --quiet omni-call-harness.service 2>/dev/null; then
  harness=--with-harness
fi

log "redeploying ${remote:0:12} with profile $profile ($harness)"
"$DEPLOY" --profile "$profile" --action deploy --yes --no-update "$harness"
git -C "$REPO_ROOT" rev-parse HEAD >"$REPO_ROOT/.deployed-commit"
log "deployed ${remote:0:12}"
