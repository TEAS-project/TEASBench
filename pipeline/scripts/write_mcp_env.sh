#!/usr/bin/env bash
#
# Materialise the MCP Atlas .env from environment variables.
#
# mcp-server/start.sh requires an .env file and exits 1 without one, then does
# `set -a && source "$ENV_FILE"` -- so .env values win over anything already
# exported into the container. Credentials therefore have to reach it through
# the file, not through the environment, which is what this script bridges: on
# Vast.ai the keys arrive as instance secrets, and this turns them into the
# file start.sh insists on.
#
# Keys are taken from the checked-out submodule's env.template rather than a
# list hardcoded here, so a new upstream key can never be silently dropped.
# For each KEY in the template, a same-named non-empty environment variable
# supplies the value; anything unset is written empty, which is what the
# template itself ships.
#
# Usage: write_mcp_env.sh <atlas_dir> [enabled_servers]
#
# NOTE: keep this heredoc-free -- its lines are re-indented when inlined into a
# YAML block scalar, which would break an unquoted heredoc terminator.
set -euo pipefail

ATLAS_DIR="${1:?usage: write_mcp_env.sh <atlas_dir> [enabled_servers]}"
ENABLED_SERVERS_ARG="${2:-}"

TEMPLATE="$ATLAS_DIR/env.template"
ENV_FILE="$ATLAS_DIR/.env"

if [ ! -f "$TEMPLATE" ]; then
    echo "ERROR: $TEMPLATE not found. The mcp-atlas submodule is not checked out;" >&2
    echo "       clone AgentCAP with --recurse-submodules." >&2
    exit 1
fi

: > "$ENV_FILE"
chmod 600 "$ENV_FILE"

populated=""
empty=""

# Read the key names straight from the template (lines like `KEY=`).
while IFS= read -r key; do
    [ -n "$key" ] || continue

    if [ "$key" = "ENABLED_SERVERS" ] && [ -n "$ENABLED_SERVERS_ARG" ]; then
        printf '%s=%s\n' "$key" "$ENABLED_SERVERS_ARG" >> "$ENV_FILE"
        continue
    fi

    # Indirect expansion: value of the env var whose name is $key.
    value="${!key:-}"
    if [ -n "$value" ]; then
        printf '%s=%s\n' "$key" "$value" >> "$ENV_FILE"
        populated="$populated $key"
    else
        printf '%s=\n' "$key" >> "$ENV_FILE"
        empty="$empty $key"
    fi
done < <(grep -oE '^[A-Z][A-Z0-9_]*=' "$TEMPLATE" | tr -d '=')

# Provenance, names only -- never values. Which credentials were present
# changes what the tool servers can actually do: mcp-atlas starts servers whose
# keys are blank and they fail only at tool-call time, so a run with missing
# credentials scores lower without failing. Recording this is what makes an
# accuracy number interpretable after the fact.
echo "MCP .env written to $ENV_FILE"
echo "  credentials supplied:$( [ -n "$populated" ] && echo "$populated" || echo " (none)" )"
echo "  left empty:$( [ -n "$empty" ] && echo "$empty" || echo " (none)" )"
