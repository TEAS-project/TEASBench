#!/usr/bin/env bash
# =============================================================================
# start_tt_server.sh
#
# Brings up a Tenstorrent inference server via tt-inference-server's run.py,
# then sends a smoke-test request to confirm the model generates.
#
# Proven for: Llama-3.1-8B-Instruct on a single Blackhole p150b card.
# Parameterised so the same flow can be reused for other model/device pairs
# listed in the Tenstorrent model-support tables — but note the weight-staging
# below assumes THIS machine's layout (network-FS home + rootless podman),
# where container-facing files must live on local /dev/shm. On a different
# machine those workarounds may need revisiting.
#
# WHY THE WORKAROUNDS EXIST (don't remove without understanding):
#   * Home is on a network filesystem (Lustre); rootless podman's UID remapping
#     can't read weights from there, so weights are staged to local /dev/shm.
#   * /dev/shm is RAM-backed and WIPED ON REBOOT, so the script re-stages
#     weights and re-points symlinks every run (idempotent — safe to re-run).
#   * The two p150b cards are ISOLATED; opening both segfaults tt-metal's
#     fabric init, so we scope to a single card via --device-id.
#   * Weights are passed to vLLM as a local snapshot PATH (not a repo id),
#     because in offline mode vLLM can't resolve a repo id.
#
# USAGE:
#   ./start_tt_server.sh                 # bring up server + smoke test
#   ./start_tt_server.sh --no-test       # bring up only
#   ./start_tt_server.sh --test-only     # only smoke-test an already-running server
#   ./start_tt_server.sh --force         # tear down a running server, relaunch fresh
#
# Config is via the block below or environment overrides, e.g.:
#   MODEL=Llama-3.1-8B-Instruct DEVICE=p150 ./start_tt_server.sh
#
# Note: `set -e` is intentionally NOT used — we want to reach the health check
# and print actionable diagnostics even if an individual step complains.
# =============================================================================

set -uo pipefail

# ---------------------------------------------------------------------------
# CONFIG  — override any of these via environment variables when invoking.
# ---------------------------------------------------------------------------
# What to run (must match a model/device pair in the Tenstorrent support tables):
MODEL="${MODEL:-Llama-3.1-8B-Instruct}"
DEVICE="${DEVICE:-p150}"
DEVICE_ID="${DEVICE_ID:-0}"                 # single ISOLATED card to scope to
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

# Where things live (defaults suit this machine; override if your layout differs):
REPO_DIR="${REPO_DIR:-$HOME/tt-inference-server}"
USER_NS="${USER_NS:-$(whoami)}"             # namespaces the /dev/shm staging dir
SHM_BASE="${SHM_BASE:-/dev/shm/${USER_NS}}"

# Repo source, used ONLY if you pass --clone and REPO_DIR doesn't exist.
# REPO_REF pins the version — a bare clone of main may behave differently, so
# set this to the tag/commit you validated against.
# Pinned to the commit where Llama-3.1-8B-Instruct was validated on p150.
REPO_URL="${REPO_URL:-https://github.com/tenstorrent/tt-inference-server.git}"
REPO_REF="${REPO_REF:-e89c4ce16}"           # validated commit; override for other versions

# Minimum free space (GB) required on /dev/shm before staging weights.
MIN_SHM_FREE_GB="${MIN_SHM_FREE_GB:-20}"

# HuggingFace model identity. HF_REPO is the on-disk cache dir name; SNAPSHOT_HASH
# pins the exact revision. Find the hash with:
#   ls ~/.cache/huggingface/hub/<HF_REPO>/snapshots/
HF_REPO="${HF_REPO:-models--meta-llama--Llama-3.1-8B-Instruct}"
SNAPSHOT_HASH="${SNAPSHOT_HASH:-0e9e39f249a16976918f6564b8830bc894c89659}"

# The HuggingFace repo id used to DOWNLOAD (e.g. meta-llama/Llama-3.1-8B-Instruct).
# Derived from HF_REPO by stripping the "models--" prefix and turning "--" into "/".
# Override if the derivation doesn't match your model.
_derived_id="${HF_REPO#models--}"; _derived_id="${_derived_id//--//}"
HF_MODEL_ID="${HF_MODEL_ID:-$_derived_id}"

# Server + auth:
SERVICE_PORT="${SERVICE_PORT:-8000}"
JWT_TEAM_ID="${JWT_TEAM_ID:-tenstorrent}"   # claims baked into the smoke-test JWT
JWT_TOKEN_ID="${JWT_TOKEN_ID:-debug-test}"

# Pinned container image — the exact build that served Llama-3.1-8B on p150.
# Empty = let run.py resolve the image from the model spec (may drift over time).
DOCKER_IMAGE="${DOCKER_IMAGE:-ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.10.0-55fd115-aa4ae1e}"

# Interpreter for run.py. Leave empty to auto-detect one that has run.py's deps
# (notably `yaml`). Set explicitly if auto-detect fails, e.g. RUN_PYTHON=/usr/bin/python3
RUN_PYTHON="${RUN_PYTHON:-}"

# ---------------------------------------------------------------------------
# Derived values (no need to edit)
# ---------------------------------------------------------------------------
HF_CACHE="${SHM_BASE}/hf_cache"
LUSTRE_HF="$HOME/.cache/huggingface/hub/${HF_REPO}"
STAGED_MODEL="${HF_CACHE}/hub/${HF_REPO}"
# Human-readable model name = repo dir with the "models--org--" prefix stripped:
MODEL_SHORT="${HF_REPO##*--}"
CONTAINER_MODEL_PATH="/home/container_app_user/readonly_weights_mount/${MODEL_SHORT}/snapshots/${SNAPSHOT_HASH}"
VENV_DIR="${REPO_DIR}/request-venv"
HEALTH_URL="http://localhost:${SERVICE_PORT}/health"
COMPLETIONS_URL="http://localhost:${SERVICE_PORT}/v1/completions"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
RUN_TEST=1; FORCE_RESTART=0; TEST_ONLY=0; DO_CLONE=0
for arg in "$@"; do
  case "$arg" in
    --no-test)   RUN_TEST=0 ;;
    --force)     FORCE_RESTART=1 ;;
    --test-only) TEST_ONLY=1 ;;
    --clone)     DO_CLONE=1 ;;   # clone the repo if REPO_DIR is missing (opt-in)
    -h|--help)   grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $arg (valid: --no-test, --force, --test-only, --clone, --help)" >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------------------
# Small helpers: coloured status lines + a health probe used throughout.
# ---------------------------------------------------------------------------
log()  { printf '\n\033[1;34m[*]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[ok]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[x]\033[0m %s\n' "$*" >&2; exit 1; }

# Interpreter for run.py (and venv creation). Populated during bring-up by
# pick_run_python; declared here so it's always defined even in --test-only mode.
SYS_PYTHON=""

server_is_healthy() {
  [[ "$(curl -s -o /dev/null -w '%{http_code}' "$HEALTH_URL" 2>/dev/null || echo 000)" == "200" ]]
}

# ---------------------------------------------------------------------------
# 0. Preconditions — fail early with a clear message rather than mid-run.
# ---------------------------------------------------------------------------
log "Checking preconditions (model=${MODEL}, device=${DEVICE}, card=${DEVICE_ID})"
command -v curl    >/dev/null || die "curl not found on PATH"
command -v docker  >/dev/null || die "docker/podman not found on PATH"
command -v tt-smi  >/dev/null || die "tt-smi not found on PATH"

# --- Repo: present, or clone it (only with --clone; never a silent side effect).
if [[ ! -d "$REPO_DIR" ]]; then
  if [[ "$DO_CLONE" == "1" ]]; then
    command -v git >/dev/null || die "git not found — needed for --clone"
    log "Cloning tt-inference-server into ${REPO_DIR}"
    git clone "$REPO_URL" "$REPO_DIR" || die "git clone failed"
    if [[ -n "$REPO_REF" ]]; then
      ( cd "$REPO_DIR" && git checkout "$REPO_REF" ) || die "could not checkout REPO_REF=${REPO_REF}"
      ok "checked out ${REPO_REF}"
    else
      warn "cloned the default branch — no REPO_REF pinned. Behaviour may differ from a validated version."
    fi
    warn "A fresh clone still needs setup (.env with JWT_SECRET/HF_TOKEN, podman storage config)."
  else
    die "repo not found at ${REPO_DIR}. Clone it first:
      git clone ${REPO_URL} ${REPO_DIR}
    then set up its .env, or re-run this script with --clone."
  fi
fi
[[ -f "${REPO_DIR}/run.py" ]] || die "run.py not found in ${REPO_DIR} — is REPO_DIR correct?"

# --- Repo version: verify it's on the validated commit. Mismatched versions
#     behave differently, so we warn (or checkout if it's a clean git tree).
if [[ -n "$REPO_REF" ]] && command -v git >/dev/null && [[ -d "${REPO_DIR}/.git" ]]; then
  CUR_REF="$(cd "$REPO_DIR" && git rev-parse --short HEAD 2>/dev/null || echo '?')"
  if [[ "$CUR_REF" != "$REPO_REF"* && "$REPO_REF" != "$CUR_REF"* ]]; then
    warn "repo is at ${CUR_REF}, expected ${REPO_REF} (the validated commit)."
    if (cd "$REPO_DIR" && git diff --quiet && git diff --cached --quiet) 2>/dev/null; then
      warn "working tree is clean — checking out ${REPO_REF}"
      (cd "$REPO_DIR" && git checkout "$REPO_REF") || die "could not checkout ${REPO_REF}"
      ok "now at ${REPO_REF}"
    else
      warn "working tree has local changes — NOT switching. Behaviour may differ from validated."
    fi
  else
    ok "repo at validated commit ${CUR_REF}"
  fi
fi

# --- Device: is a Tenstorrent card actually visible? Catches unloaded drivers
#     or an empty slot before we commit to a long launch.
if ! tt-smi -s >/dev/null 2>&1; then
  warn "tt-smi could not read device telemetry — the card may not be visible."
  warn "Check the driver is loaded and a card is present before relying on this run."
fi

# --- .env: the server reads it for JWT_SECRET and HF_TOKEN. Bootstrap a minimal
#     one if missing (with JWT_SECRET and a placeholder token), then validate.
ENV_FILE="${REPO_DIR}/.env"
if [[ ! -f "$ENV_FILE" ]]; then
  warn "no .env found — creating a minimal one at ${ENV_FILE}"
  {
    echo "JWT_SECRET=${JWT_SECRET:-testing}"
    echo "HF_TOKEN=${HF_TOKEN:-hf_REPLACE_WITH_YOUR_TOKEN}"
  } > "$ENV_FILE"
  ok "wrote minimal .env (JWT_SECRET + HF_TOKEN placeholder)"
fi
# Validate the two keys are present and the HF token isn't the placeholder.
grep -q '^JWT_SECRET=' "$ENV_FILE" || warn ".env has no JWT_SECRET — requests may 401"
if ! grep -q '^HF_TOKEN=' "$ENV_FILE"; then
  warn ".env has no HF_TOKEN — model download will fail if weights aren't already present"
elif grep -q '^HF_TOKEN=hf_REPLACE_WITH_YOUR_TOKEN' "$ENV_FILE"; then
  warn ".env HF_TOKEN is still the placeholder — edit ${ENV_FILE} and set a valid token if weights need downloading"
fi
# These offline flags must NOT be pre-set here — the script adds them later, at
# the right moment. Set early they block the initial weight download.
if grep -qE '^(HF_HUB_OFFLINE|TRANSFORMERS_OFFLINE)=' "$ENV_FILE"; then
  warn ".env already sets offline flags — if you still need to DOWNLOAD weights, remove them first"
fi

# --- Podman/Docker storage: on this box, image storage MUST live on local disk
#     (not the network-FS home), or rootless containers break. We verify the
#     config exists and points at local disk; if it's missing we OFFER to write
#     the known-good version (never silently — it's host config on a shared box).
STORAGE_CONF="${HOME}/.config/containers/storage.conf"
if command -v podman >/dev/null; then
  if [[ ! -f "$STORAGE_CONF" ]]; then
    warn "no podman storage.conf at ${STORAGE_CONF}"
    warn "on a network-FS home, rootless podman needs graphroot on local disk (/dev/shm)."
    read -r -p "Write a known-good storage.conf pointing at ${SHM_BASE}/containers? [y/N] " reply
    if [[ "$reply" =~ ^[Yy]$ ]]; then
      mkdir -p "$(dirname "$STORAGE_CONF")" "${SHM_BASE}/containers/storage" "${SHM_BASE}/containers/run"
      cat > "$STORAGE_CONF" <<EOF
[storage]
driver = "overlay"
graphroot = "${SHM_BASE}/containers/storage"
runroot = "${SHM_BASE}/containers/run"
[storage.options.overlay]
mount_program = "/usr/bin/fuse-overlayfs"
force_mask = "1777"
ignore_chown_errors = "true"
EOF
      ok "wrote ${STORAGE_CONF} (graphroot on ${SHM_BASE})"
      warn "NOTE: /dev/shm is wiped on reboot, so images re-pull after a restart. Run 'podman system migrate' if podman complains."
    else
      warn "skipped — set up storage.conf yourself before the launch, or containers may fail"
    fi
  else
    # Config exists — sanity-check graphroot is on local disk, not \$HOME.
    GRAPHROOT="$(podman info --format '{{.Store.GraphRoot}}' 2>/dev/null || echo '')"
    if [[ -n "$GRAPHROOT" && "$GRAPHROOT" == "$HOME"* && "$GRAPHROOT" != /dev/shm/* ]]; then
      warn "podman graphRoot is under \$HOME (${GRAPHROOT}) — on a network-FS home this can break rootless containers. Check ${STORAGE_CONF}."
    else
      ok "podman storage looks fine (graphroot: ${GRAPHROOT:-unknown})"
    fi
  fi
fi

# --- HF token (needed only if weights must be downloaded; offline runs don't).
[[ -n "${HF_TOKEN:-}" ]] || warn "HF_TOKEN not set (fine if weights are already downloaded and you run offline)."

# JWT secret must match what the server was started with, or requests get 401.
# Prefer the shell value; else read it from the repo .env (which the server reads).
if [[ -z "${JWT_SECRET:-}" ]]; then
  if [[ -f "${REPO_DIR}/.env" ]] && grep -q '^JWT_SECRET=' "${REPO_DIR}/.env"; then
    # Strip only the leading 'JWT_SECRET=' so values containing '=' survive intact.
    JWT_SECRET="$(grep '^JWT_SECRET=' "${REPO_DIR}/.env" | head -1 | sed 's/^JWT_SECRET=//')"; export JWT_SECRET
    ok "JWT_SECRET loaded from .env"
  else
    export JWT_SECRET="testing"
    warn "JWT_SECRET not set — defaulting to 'testing' (must match the server's secret)"
  fi
fi

# ---------------------------------------------------------------------------
# 1. Preflight — is a server already running? Decide reuse vs. relaunch.
#    This is what makes the script safe to run repeatedly / against a live server.
# ---------------------------------------------------------------------------
log "Preflight: checking for an existing server on port ${SERVICE_PORT}"
# Locate the serving container. Podman rejects Docker's 'publish' filter, so we
# match by the tt-inference-server name prefix, then fall back to a ports scan.
find_server_container() {
  local cid
  cid="$(docker ps -q --filter 'name=tt-inference-server' 2>/dev/null | head -n1)"
  [[ -n "$cid" ]] && { echo "$cid"; return; }
  docker ps --format '{{.ID}} {{.Ports}}' 2>/dev/null | awk -v p="${SERVICE_PORT}" '$0 ~ p {print $1; exit}'
}
RUNNING_CID="$(find_server_container)"

if server_is_healthy; then
  warn "A healthy server is already running at ${HEALTH_URL}"
  if [[ "$FORCE_RESTART" == "1" ]]; then
    warn "--force: stopping it and relaunching from scratch"
    if [[ -n "$RUNNING_CID" ]]; then
      docker stop "$RUNNING_CID" >/dev/null 2>&1 && ok "stopped container $RUNNING_CID" \
        || warn "docker stop returned non-zero"
    else
      warn "server is healthy but no matching container found — you may need to stop it by hand"
    fi
    # Wait for the port to genuinely free, so the later health poll can't be
    # fooled by the old server still answering (a false-healthy race).
    DOWN=0
    for _ in $(seq 1 30); do server_is_healthy || { DOWN=1; break; }; sleep 1; done
    [[ "$DOWN" == "1" ]] || die "old server still answering on ${SERVICE_PORT} after stop — stop it manually (docker ps; docker stop <id>) and re-run"
    ok "old server confirmed down"
  else
    ok "Reusing the running server (skipping bring-up). Use --force to relaunch."
    TEST_ONLY=1
  fi
elif [[ "$TEST_ONLY" == "1" ]]; then
  die "--test-only given but no healthy server at ${HEALTH_URL}. Bring one up first."
elif [[ -n "$RUNNING_CID" ]]; then
  # A container holds the port but isn't healthy — clear it to avoid a port clash.
  warn "Container $RUNNING_CID holds port ${SERVICE_PORT} but isn't healthy — stopping it"
  docker stop "$RUNNING_CID" >/dev/null 2>&1 && ok "stopped stale container $RUNNING_CID"
  sleep 3
fi

# Decide whether to run the bring-up (steps 2-5) or jump straight to the test.
if [[ "$TEST_ONLY" == "1" ]]; then RUN_TEST=1; SKIP_BRINGUP=1; else SKIP_BRINGUP=0; fi

if [[ "$SKIP_BRINGUP" == "0" ]]; then

  # -------------------------------------------------------------------------
  # 2. Ensure weights exist in the permanent HF cache (download once if not),
  #    then stage them on local /dev/shm for the container.
  #    Rootless podman can't read the network-FS home, so the container reads
  #    weights from /dev/shm; cp -a preserves the snapshot->blob symlinks that
  #    HuggingFace needs in offline mode.
  # -------------------------------------------------------------------------
  # 2a. Download to the permanent HF cache if the model isn't there yet.
  if [[ -d "$LUSTRE_HF" ]]; then
    ok "Weights present in HF cache (${HF_MODEL_ID})"
  else
    warn "Weights not in HF cache — downloading ${HF_MODEL_ID} (one-time, ~several GB)"
    # Pick a downloader: the modern 'hf' CLI or the older 'huggingface-cli'.
    HF_CLI=""
    command -v hf >/dev/null && HF_CLI="hf download"
    [[ -z "$HF_CLI" ]] && command -v huggingface-cli >/dev/null && HF_CLI="huggingface-cli download"
    [[ -n "$HF_CLI" ]] || die "no HuggingFace CLI found (need 'hf' or 'huggingface-cli'). Install it, or download ${HF_MODEL_ID} manually."
    # A token is needed for gated models (e.g. Llama). Prefer shell, else .env.
    DL_TOKEN="${HF_TOKEN:-}"
    [[ -z "$DL_TOKEN" && -f "$ENV_FILE" ]] && DL_TOKEN="$(grep '^HF_TOKEN=' "$ENV_FILE" | head -1 | sed 's/^HF_TOKEN=//')"
    [[ "$DL_TOKEN" == "hf_REPLACE_WITH_YOUR_TOKEN" ]] && DL_TOKEN=""
    [[ -n "$DL_TOKEN" ]] || warn "no valid HF_TOKEN — download will fail if ${HF_MODEL_ID} is gated"
    HF_TOKEN="$DL_TOKEN" $HF_CLI "$HF_MODEL_ID" \
      || die "weight download failed for ${HF_MODEL_ID} — check the token has access and the id is correct"
    [[ -d "$LUSTRE_HF" ]] || die "download reported success but ${LUSTRE_HF} not found — check HF_REPO matches ${HF_MODEL_ID}"
    ok "Downloaded ${HF_MODEL_ID}"
  fi

  # 2b. Stage from the HF cache onto local /dev/shm.
  log "Staging weights on local disk (${HF_CACHE})"
  if [[ -d "${STAGED_MODEL}/snapshots/${SNAPSHOT_HASH}" ]]; then
    ok "Weights already staged"
  else
    warn "Weights not staged (fresh boot?) — copying from ${LUSTRE_HF}"
    # Disk preflight: fail clearly now rather than mid-copy on a near-full /dev/shm.
    SHM_FREE_GB="$(df -BG --output=avail /dev/shm 2>/dev/null | tail -1 | tr -dc '0-9')"
    if [[ -n "$SHM_FREE_GB" && "$SHM_FREE_GB" -lt "$MIN_SHM_FREE_GB" ]]; then
      die "only ${SHM_FREE_GB}GB free on /dev/shm, need ~${MIN_SHM_FREE_GB}GB. Free space (see cleanup_tt.sh) and retry."
    fi
    mkdir -p "${HF_CACHE}/hub" || die "cannot create ${HF_CACHE}/hub"
    cp -a "$LUSTRE_HF" "${HF_CACHE}/hub/" || die "weight copy failed — check disk: df -h /dev/shm"
    [[ -f "${STAGED_MODEL}/snapshots/${SNAPSHOT_HASH}/config.json" ]] \
      || die "staged weights look incomplete (no config.json at the pinned snapshot) — check SNAPSHOT_HASH"
    ok "Weights staged"
  fi

  # -------------------------------------------------------------------------
  # 3. Point HuggingFace at the local cache and force offline mode, so the
  #    container uses the staged weights and never tries to reach the network.
  #    The flags are also written to .env so they reach the container too.
  # -------------------------------------------------------------------------
  log "Configuring offline HuggingFace environment"
  export HF_HOME="${HF_CACHE}"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  for kv in "HF_HUB_OFFLINE=1" "TRANSFORMERS_OFFLINE=1"; do
    grep -q "^${kv%%=*}=" "${REPO_DIR}/.env" 2>/dev/null || { echo "$kv" >> "${REPO_DIR}/.env" && ok "added ${kv%%=*} to .env"; }
  done

  # -------------------------------------------------------------------------
  # 4. Re-point the local-disk symlinks run.py expects. These go stale after a
  #    reboot (their /dev/shm targets vanish), which otherwise causes a
  #    FileExistsError on launch. rm -rf clears a stale link or a real dir.
  # -------------------------------------------------------------------------
  log "Re-pointing local-disk symlinks"
  relink() {  # $1 = repo-relative link path, $2 = /dev/shm target
    local link="${REPO_DIR}/$1" target="$2"
    mkdir -p "$target"; rm -rf "$link"; mkdir -p "$(dirname "$link")"
    ln -s "$target" "$link" || die "failed to symlink $link -> $target"
    ok "$1 -> $target"
  }
  relink "workflow_logs/run_specs" "${SHM_BASE}/run_specs"
  relink "persistent_volume"       "${SHM_BASE}/persistent_volume"

  # -------------------------------------------------------------------------
  # 5. Reset the card, then launch the server.
  #    run.py needs deps (e.g. yaml) that the pyjwt-only request venv lacks, so
  #    we pick an interpreter that can import yaml rather than trusting whatever
  #    `python3` resolves to (which breaks if launched from an active venv).
  # -------------------------------------------------------------------------
  log "Resetting Tenstorrent card ${DEVICE_ID}"
  tt-smi -r || warn "tt-smi -r returned non-zero — continuing (check card state if launch fails)"

  pick_run_python() {
    local candidates=("/usr/bin/python3" "/usr/local/bin/python3" "/opt/tenstorrent/.tenstorrent-venv/bin/python3")
    local pathpy; pathpy="$(command -v python3 || true)"
    [[ "$pathpy" == *"request-venv"* ]] || candidates+=("$pathpy")
    for py in "${candidates[@]}"; do
      [[ -x "$py" ]] || continue
      "$py" -c 'import yaml' 2>/dev/null && { echo "$py"; return 0; }
    done
    return 1
  }
  SYS_PYTHON="${RUN_PYTHON:-$(pick_run_python)}" \
    || die "no python found with run.py's deps (yaml). Run from a plain shell (deactivate any venv first) or set RUN_PYTHON"
  "$SYS_PYTHON" -c 'import yaml' 2>/dev/null \
    || die "interpreter ${SYS_PYTHON} can't import yaml — set RUN_PYTHON to one with tt-inference-server's deps"
  ok "Using interpreter: ${SYS_PYTHON}"

  log "Launching server (first run compiles a cache — can take several minutes)"
  cd "$REPO_DIR" || die "cannot cd to $REPO_DIR"
  # Build the optional image override only when DOCKER_IMAGE is set, so an empty
  # value cleanly falls back to run.py's spec-resolved image.
  IMAGE_ARGS=()
  [[ -n "$DOCKER_IMAGE" ]] && IMAGE_ARGS=(--override-docker-image "$DOCKER_IMAGE")
  [[ -n "$DOCKER_IMAGE" ]] && ok "pinned image: ${DOCKER_IMAGE}"
  "$SYS_PYTHON" run.py --model "$MODEL" --device "$DEVICE" --workflow server --docker-server \
    --device-id "$DEVICE_ID" \
    "${IMAGE_ARGS[@]}" \
    --vllm-override-args "{\"max_model_len\": ${MAX_MODEL_LEN}, \"model\": \"${CONTAINER_MODEL_PATH}\"}" \
    &
  warn "run.py launched (PID $!). Its own 30s startup timeout can fire spuriously;"
  warn "we poll /health directly below instead of trusting run.py's exit."

fi  # end bring-up

# ---------------------------------------------------------------------------
# 6. Wait for health directly (run.py's own readiness signal is unreliable here).
# ---------------------------------------------------------------------------
log "Waiting for the server to become healthy at ${HEALTH_URL} (up to ~10 min)"
HEALTHY=0
for i in $(seq 1 120); do   # 120 * 5s = 600s
  if server_is_healthy; then HEALTHY=1; ok "Server healthy after ~$((i*5))s"; break; fi
  sleep 5
done
[[ "$HEALTHY" == "1" ]] || die "server did not become healthy — inspect the newest log under ${REPO_DIR}/workflow_logs/docker_server/"

# ---------------------------------------------------------------------------
# 7. Smoke test (unless --no-test). Confirms the model actually generates.
# ---------------------------------------------------------------------------
if [[ "$RUN_TEST" == "0" ]]; then
  ok "Server is up. Skipping smoke test (--no-test)."
  exit 0
fi

# The request needs a JWT signed with JWT_SECRET. pyjwt lives in the request venv;
# we call that venv's python DIRECTLY (no `source activate`) so nothing leaks into
# the surrounding environment.
log "Preparing API token"
# In --test-only mode the bring-up block was skipped, so SYS_PYTHON may be empty.
# Fall back to any python3 for the (rare) case where the venv needs creating.
VENV_PYTHON="${SYS_PYTHON:-$(command -v python3)}"
if [[ ! -d "$VENV_DIR" ]]; then
  warn "request venv missing — creating it"
  "$VENV_PYTHON" -m venv "$VENV_DIR" || die "venv creation failed"
  "${VENV_DIR}/bin/pip" install --quiet pyjwt==2.7.0 || die "pip install pyjwt failed"
fi
"${VENV_DIR}/bin/python" -c 'import jwt' 2>/dev/null \
  || "${VENV_DIR}/bin/pip" install --quiet pyjwt==2.7.0 || die "pyjwt unavailable in request venv"

VLLM_API_KEY="$(JWT_SECRET="$JWT_SECRET" TEAM="$JWT_TEAM_ID" TID="$JWT_TOKEN_ID" \
  "${VENV_DIR}/bin/python" -c 'import os,jwt; print(jwt.encode({"team_id":os.environ["TEAM"],"token_id":os.environ["TID"]}, os.environ["JWT_SECRET"], algorithm="HS256"))')"
[[ -n "$VLLM_API_KEY" ]] || die "token generation produced an empty key (would cause 401) — check the request venv and JWT_SECRET"
ok "API token generated (length ${#VLLM_API_KEY})"

log "Sending smoke-test request (first request is slow ~60-80s: lazy warmup, not a hang)"
RESP="$(curl -sS "$COMPLETIONS_URL" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${VLLM_API_KEY}" \
  -d "{\"model\": \"${CONTAINER_MODEL_PATH}\", \"prompt\": \"San Francisco is a\", \"max_tokens\": 50, \"temperature\": 0}")"

echo "$RESP" | python3 -m json.tool 2>/dev/null || echo "$RESP"

if echo "$RESP" | grep -q '"text"'; then
  ok "SUCCESS — ${MODEL_SHORT} generated text on ${DEVICE}."
elif echo "$RESP" | grep -qi 'unauthorized'; then
  die "401 Unauthorized — JWT_SECRET mismatch or empty token. Confirm it matches the server's secret."
else
  warn "Request returned but no completion text found — inspect the response above."
fi