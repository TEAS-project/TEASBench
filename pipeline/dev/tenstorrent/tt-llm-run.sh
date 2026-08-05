#!/usr/bin/env bash
# tt-llm-run.sh v1.1.0 — Tenstorrent LLM run
# Brings up a tt-inference-server vLLM server and smoke-tests it.
# Validated: Llama-3.1-8B-Instruct on a single Blackhole p150.
# Pinned: repo v0.10.0 + image 0.10.0-55fd115-aa4ae1e (must stay a matched pair).
#
# Usage:
#   ./tt-llm-run.sh [--clone] [--force] [--test-only]
#     --clone      clone the repo into REPO_DIR if missing
#     --force      tear down a running server and relaunch
#     --test-only  smoke-test an already-running server
#   Override any CONFIG value via env, e.g. MODEL=... DEVICE=... ./tt-llm-run.sh

set -uo pipefail

# ---- CONFIG (override via env) ----
MODEL="${MODEL:-Llama-3.1-8B-Instruct}"
DEVICE="${DEVICE:-p150}"
DEVICE_ID="${DEVICE_ID:-0}"                 # single ISOLATED card
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

REPO_DIR="${REPO_DIR:-$HOME/tt-inference-server}"
USER_NS="${USER_NS:-$(whoami)}"
SHM_BASE="${SHM_BASE:-/dev/shm/${USER_NS}}"
MIN_SHM_FREE_GB="${MIN_SHM_FREE_GB:-20}"

REPO_URL="${REPO_URL:-https://github.com/tenstorrent/tt-inference-server.git}"
REPO_REF="${REPO_REF:-v0.10.0}"             # matched pair with DOCKER_IMAGE
DOCKER_IMAGE="${DOCKER_IMAGE:-ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.10.0-55fd115-aa4ae1e}"

HF_REPO="${HF_REPO:-models--meta-llama--Llama-3.1-8B-Instruct}"
SNAPSHOT_HASH="${SNAPSHOT_HASH:-0e9e39f249a16976918f6564b8830bc894c89659}"
_derived_id="${HF_REPO#models--}"; _derived_id="${_derived_id//--//}"
HF_MODEL_ID="${HF_MODEL_ID:-$_derived_id}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$HF_MODEL_ID}"

SERVICE_PORT="${SERVICE_PORT:-8000}"
JWT_TEAM_ID="${JWT_TEAM_ID:-tenstorrent}"
JWT_TOKEN_ID="${JWT_TOKEN_ID:-debug-test}"
DISABLE_TRACE_CAPTURE="${DISABLE_TRACE_CAPTURE:-0}"
RUN_PYTHON="${RUN_PYTHON:-}"               # empty = auto-detect a yaml-capable python

# ---- derived ----
HF_CACHE="${SHM_BASE}/hf_cache"
LUSTRE_HF="$HOME/.cache/huggingface/hub/${HF_REPO}"
STAGED_MODEL="${HF_CACHE}/hub/${HF_REPO}"
MODEL_SHORT="${HF_REPO##*--}"
CONTAINER_MODEL_PATH="/home/container_app_user/readonly_weights_mount/${MODEL_SHORT}/snapshots/${SNAPSHOT_HASH}"
VENV_DIR="${REPO_DIR}/request-venv"
ENV_FILE="${REPO_DIR}/.env"
HEALTH_URL="http://localhost:${SERVICE_PORT}/health"
COMPLETIONS_URL="http://localhost:${SERVICE_PORT}/v1/completions"

# ---- args ----
FORCE_RESTART=0; TEST_ONLY=0; DO_CLONE=0
for arg in "$@"; do
  case "$arg" in
    --force)     FORCE_RESTART=1 ;;
    --test-only) TEST_ONLY=1 ;;
    --clone)     DO_CLONE=1 ;;
    -h|--help)   sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

log()  { printf '\n\033[1;34m[*]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m[ok]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[x]\033[0m %s\n' "$*" >&2; exit 1; }
server_is_healthy() { [[ "$(curl -s -o /dev/null -w '%{http_code}' "$HEALTH_URL" 2>/dev/null || echo 000)" == "200" ]]; }

SYS_PYTHON=""

# ---- 0. preconditions ----
log "Preconditions (model=${MODEL}, device=${DEVICE}, card=${DEVICE_ID})"
command -v curl   >/dev/null || die "curl not found"
command -v docker >/dev/null || die "docker/podman not found"
command -v tt-smi >/dev/null || die "tt-smi not found"

if [[ ! -d "$REPO_DIR" ]]; then
  [[ "$DO_CLONE" == "1" ]] || die "repo not found at ${REPO_DIR}. Clone it or re-run with --clone."
  command -v git >/dev/null || die "git not found (needed for --clone)"
  log "Cloning into ${REPO_DIR}"
  git clone "$REPO_URL" "$REPO_DIR" || die "git clone failed"
  [[ -n "$REPO_REF" ]] && { ( cd "$REPO_DIR" && git checkout "$REPO_REF" ) || die "checkout ${REPO_REF} failed"; }
fi
[[ -f "${REPO_DIR}/run.py" ]] || die "run.py not found in ${REPO_DIR}"

# pin repo to REPO_REF if the tree is clean
if [[ -n "$REPO_REF" ]] && command -v git >/dev/null && [[ -d "${REPO_DIR}/.git" ]]; then
  CUR_REF="$(cd "$REPO_DIR" && git rev-parse --short HEAD 2>/dev/null || echo '?')"
  if [[ "$CUR_REF" != "$REPO_REF"* && "$REPO_REF" != "$CUR_REF"* ]]; then
    if (cd "$REPO_DIR" && git diff --quiet && git diff --cached --quiet) 2>/dev/null; then
      (cd "$REPO_DIR" && git checkout "$REPO_REF") || die "checkout ${REPO_REF} failed"; ok "repo at ${REPO_REF}"
    else
      warn "repo at ${CUR_REF} with local changes — NOT switching to ${REPO_REF}"
    fi
  fi
fi

tt-smi -s >/dev/null 2>&1 || warn "tt-smi can't read the card — check the driver/card"

# .env: bootstrap minimal, then validate
if [[ ! -f "$ENV_FILE" ]]; then
  { echo "JWT_SECRET=${JWT_SECRET:-testing}"; echo "HF_TOKEN=${HF_TOKEN:-hf_REPLACE_WITH_YOUR_TOKEN}"; } > "$ENV_FILE"
  ok "wrote minimal .env"
fi
grep -q '^JWT_SECRET=' "$ENV_FILE" || warn ".env has no JWT_SECRET (requests may 401)"
grep -q '^HF_TOKEN=hf_REPLACE_WITH_YOUR_TOKEN' "$ENV_FILE" && warn ".env HF_TOKEN is a placeholder — set a real token if weights need downloading"
grep -qE '^(HF_HUB_OFFLINE|TRANSFORMERS_OFFLINE)=' "$ENV_FILE" && warn ".env sets offline flags — remove them if you still need to download weights"

# podman storage must be on local disk; offer to write config if missing
STORAGE_CONF="${HOME}/.config/containers/storage.conf"
if command -v podman >/dev/null; then
  if [[ ! -f "$STORAGE_CONF" ]]; then
    warn "no podman storage.conf — rootless podman needs graphroot on local disk"
    read -r -p "Write one pointing at ${SHM_BASE}/containers? [y/N] " reply
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
      ok "wrote ${STORAGE_CONF}"
    else
      warn "skipped — containers may fail without it"
    fi
  else
    GRAPHROOT="$(podman info --format '{{.Store.GraphRoot}}' 2>/dev/null || echo '')"
    [[ -n "$GRAPHROOT" && "$GRAPHROOT" == "$HOME"* && "$GRAPHROOT" != /dev/shm/* ]] \
      && warn "podman graphRoot under \$HOME (${GRAPHROOT}) — may break rootless containers"
  fi
fi

# JWT secret (for the smoke-test token); must match the server's
if [[ -z "${JWT_SECRET:-}" ]]; then
  if grep -q '^JWT_SECRET=' "$ENV_FILE" 2>/dev/null; then
    JWT_SECRET="$(grep '^JWT_SECRET=' "$ENV_FILE" | head -1 | sed 's/^JWT_SECRET=//')"; export JWT_SECRET
  else
    export JWT_SECRET="testing"; warn "JWT_SECRET defaulting to 'testing'"
  fi
fi

# ---- 1. preflight: reuse / relaunch ----
log "Preflight: existing server on port ${SERVICE_PORT}?"
find_server_container() {
  local cid; cid="$(docker ps -q --filter 'name=tt-inference-server' 2>/dev/null | head -n1)"
  [[ -n "$cid" ]] && { echo "$cid"; return; }
  docker ps --format '{{.ID}} {{.Ports}}' 2>/dev/null | awk -v p="${SERVICE_PORT}" '$0 ~ p {print $1; exit}'
}
RUNNING_CID="$(find_server_container)"

if server_is_healthy; then
  if [[ "$FORCE_RESTART" == "1" ]]; then
    [[ -n "$RUNNING_CID" ]] && { docker stop "$RUNNING_CID" >/dev/null 2>&1 && ok "stopped $RUNNING_CID"; } || warn "no matching container found"
    DOWN=0; for _ in $(seq 1 30); do server_is_healthy || { DOWN=1; break; }; sleep 1; done
    [[ "$DOWN" == "1" ]] || die "old server still on ${SERVICE_PORT} — stop it manually and re-run"
  else
    ok "reusing running server (use --force to relaunch)"; TEST_ONLY=1
  fi
elif [[ "$TEST_ONLY" == "1" ]]; then
  die "--test-only but no healthy server at ${HEALTH_URL}"
elif [[ -n "$RUNNING_CID" ]]; then
  warn "stale container on port ${SERVICE_PORT} — stopping"; docker stop "$RUNNING_CID" >/dev/null 2>&1; sleep 3
fi

[[ "$TEST_ONLY" == "1" ]] && SKIP_BRINGUP=1 || SKIP_BRINGUP=0

if [[ "$SKIP_BRINGUP" == "0" ]]; then

  # ---- 2a. download weights to HF cache if absent ----
  if [[ ! -d "$LUSTRE_HF" ]]; then
    log "Downloading ${HF_MODEL_ID} (one-time)"
    DL_TOKEN="${HF_TOKEN:-}"
    [[ -z "$DL_TOKEN" && -f "$ENV_FILE" ]] && DL_TOKEN="$(grep '^HF_TOKEN=' "$ENV_FILE" | head -1 | sed 's/^HF_TOKEN=//')"
    [[ "$DL_TOKEN" == "hf_REPLACE_WITH_YOUR_TOKEN" ]] && DL_TOKEN=""
    if command -v hf >/dev/null; then
      HF_TOKEN="$DL_TOKEN" hf download "$HF_MODEL_ID" || die "download failed"
    elif command -v huggingface-cli >/dev/null; then
      HF_TOKEN="$DL_TOKEN" huggingface-cli download "$HF_MODEL_ID" || die "download failed"
    else
      DL_PY=""
      for _py in "$(command -v python3)" /usr/bin/python3 /opt/tenstorrent/.tenstorrent-venv/bin/python3 /usr/local/bin/python3; do
        [[ -x "$_py" ]] && "$_py" -c 'import huggingface_hub' 2>/dev/null && { DL_PY="$_py"; break; }
      done
      [[ -n "$DL_PY" ]] || die "no HF downloader found. Install it and re-run: pip install --user huggingface_hub"
      HF_TOKEN="$DL_TOKEN" HF_MODEL_ID="$HF_MODEL_ID" "$DL_PY" - <<'PY' || die "download failed"
import os
from huggingface_hub import snapshot_download
snapshot_download(os.environ["HF_MODEL_ID"], token=os.environ.get("HF_TOKEN") or None)
PY
    fi
    [[ -d "$LUSTRE_HF" ]] || die "download ok but ${LUSTRE_HF} missing — check HF_REPO matches ${HF_MODEL_ID}"
    ok "downloaded ${HF_MODEL_ID}"
  fi

  # ---- 2b. stage weights onto /dev/shm (cp -a preserves snapshot->blob symlinks) ----
  if [[ ! -d "${STAGED_MODEL}/snapshots/${SNAPSHOT_HASH}" ]]; then
    log "Staging weights (${HF_CACHE})"
    SHM_FREE_GB="$(df -BG --output=avail /dev/shm 2>/dev/null | tail -1 | tr -dc '0-9')"
    [[ -n "$SHM_FREE_GB" && "$SHM_FREE_GB" -lt "$MIN_SHM_FREE_GB" ]] && die "only ${SHM_FREE_GB}GB free on /dev/shm, need ~${MIN_SHM_FREE_GB}GB"
    mkdir -p "${HF_CACHE}/hub" || die "cannot create ${HF_CACHE}/hub"
    cp -a "$LUSTRE_HF" "${HF_CACHE}/hub/" || die "copy failed — check df -h /dev/shm"
    [[ -f "${STAGED_MODEL}/snapshots/${SNAPSHOT_HASH}/config.json" ]] || die "staged weights incomplete — check SNAPSHOT_HASH"
  fi

  # ---- 3. offline HF env (also into .env so the container sees it) ----
  export HF_HOME="${HF_CACHE}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  for kv in "HF_HUB_OFFLINE=1" "TRANSFORMERS_OFFLINE=1"; do
    grep -q "^${kv%%=*}=" "$ENV_FILE" 2>/dev/null || echo "$kv" >> "$ENV_FILE"
  done

  # ---- 4. re-point /dev/shm symlinks (stale after reboot -> FileExistsError) ----
  relink() { local link="${REPO_DIR}/$1" target="$2"; mkdir -p "$target"; rm -rf "$link"; mkdir -p "$(dirname "$link")"; ln -s "$target" "$link" || die "symlink $link failed"; }
  relink "workflow_logs/run_specs" "${SHM_BASE}/run_specs"
  relink "persistent_volume"       "${SHM_BASE}/persistent_volume"

  # ---- 5. reset card + launch ----
  log "Resetting card ${DEVICE_ID}"
  tt-smi -r || warn "tt-smi -r non-zero — continuing"

  pick_run_python() {
    local candidates=("/usr/bin/python3" "/usr/local/bin/python3" "/opt/tenstorrent/.tenstorrent-venv/bin/python3")
    local pathpy; pathpy="$(command -v python3 || true)"
    [[ "$pathpy" == *"request-venv"* ]] || candidates+=("$pathpy")
    for py in "${candidates[@]}"; do
      [[ -x "$py" ]] && "$py" -c 'import yaml' 2>/dev/null && { echo "$py"; return 0; }
    done
    return 1
  }
  SYS_PYTHON="${RUN_PYTHON:-$(pick_run_python)}" || die "no python with run.py deps (yaml). Run from a plain shell or set RUN_PYTHON"

  log "Launching (first run compiles a cache — several minutes)"
  cd "$REPO_DIR" || die "cannot cd to $REPO_DIR"
  IMAGE_ARGS=(); [[ -n "$DOCKER_IMAGE" ]] && IMAGE_ARGS=(--override-docker-image "$DOCKER_IMAGE")
  TRACE_ARGS=(); [[ "$DISABLE_TRACE_CAPTURE" == "1" ]] && TRACE_ARGS=(--disable-trace-capture)
  "$SYS_PYTHON" run.py --model "$MODEL" --device "$DEVICE" --workflow server --docker-server \
    --device-id "$DEVICE_ID" "${IMAGE_ARGS[@]}" "${TRACE_ARGS[@]}" \
    --vllm-override-args "{\"max_model_len\": ${MAX_MODEL_LEN}, \"model\": \"${CONTAINER_MODEL_PATH}\", \"served_model_name\": \"${SERVED_MODEL_NAME}\"}" &
  warn "run.py launched (PID $!); polling /health directly"
fi

# ---- 6. wait for health ----
log "Waiting for health at ${HEALTH_URL} (up to ~10 min)"
HEALTHY=0
for i in $(seq 1 120); do server_is_healthy && { HEALTHY=1; ok "healthy after ~$((i*5))s"; break; }; sleep 5; done
[[ "$HEALTHY" == "1" ]] || die "server not healthy — see ${REPO_DIR}/workflow_logs/docker_server/"

# ---- 7. smoke test ----
VENV_PYTHON="${SYS_PYTHON:-$(command -v python3)}"
if [[ ! -d "$VENV_DIR" ]]; then
  "$VENV_PYTHON" -m venv "$VENV_DIR" || die "venv creation failed"
  "${VENV_DIR}/bin/pip" install --quiet pyjwt==2.7.0 || die "pip install pyjwt failed"
fi
"${VENV_DIR}/bin/python" -c 'import jwt' 2>/dev/null || "${VENV_DIR}/bin/pip" install --quiet pyjwt==2.7.0 || die "pyjwt unavailable"

VLLM_API_KEY="$(JWT_SECRET="$JWT_SECRET" TEAM="$JWT_TEAM_ID" TID="$JWT_TOKEN_ID" \
  "${VENV_DIR}/bin/python" -c 'import os,jwt; print(jwt.encode({"team_id":os.environ["TEAM"],"token_id":os.environ["TID"]}, os.environ["JWT_SECRET"], algorithm="HS256"))')"
[[ -n "$VLLM_API_KEY" ]] || die "empty token (would 401) — check request venv and JWT_SECRET"

log "Smoke test (first request slow ~60-80s: lazy warmup)"
RESP="$(curl -sS "$COMPLETIONS_URL" -H "Content-Type: application/json" -H "Authorization: Bearer ${VLLM_API_KEY}" \
  -d "{\"model\": \"${SERVED_MODEL_NAME}\", \"prompt\": \"San Francisco is a\", \"max_tokens\": 50, \"temperature\": 0}")"
echo "$RESP" | python3 -m json.tool 2>/dev/null || echo "$RESP"

if echo "$RESP" | grep -q '"text"'; then
  ok "SUCCESS — ${MODEL_SHORT} generated on ${DEVICE}."
  cat <<EOF

--- send your own requests ---
export JWT_SECRET="\$(grep '^JWT_SECRET=' ${ENV_FILE} | sed 's/^JWT_SECRET=//')"
export VLLM_API_KEY="\$('${VENV_DIR}/bin/python' -c 'import os,jwt; print(jwt.encode({"team_id":"${JWT_TEAM_ID}","token_id":"${JWT_TOKEN_ID}"}, os.environ["JWT_SECRET"], algorithm="HS256"))')"
curl -sS "${COMPLETIONS_URL}" -H "Content-Type: application/json" -H "Authorization: Bearer \$VLLM_API_KEY" \\
  -d '{"model": "${SERVED_MODEL_NAME}", "prompt": "San Francisco is a", "max_tokens": 50, "temperature": 0}' | jq
EOF
elif echo "$RESP" | grep -qi 'unauthorized'; then
  die "401 — empty/mismatched token, or wrong model name (must be '${SERVED_MODEL_NAME}')"
else
  warn "no completion text — inspect the response above"
fi
