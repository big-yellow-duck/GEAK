#!/usr/bin/env bash
# Install Codex CLI for an ephemeral container and probe inherited ChatGPT auth.
# CODEX_HOME should be a persistent/mounted directory containing auth.json from
# "codex login"; GEAK never copies or prints that credential.
set -euo pipefail

log() { printf '[setup-codex] %s\n' "$*"; }
die() { printf '[setup-codex] ERROR: %s\n' "$*" >&2; exit 1; }

: "${CODEX_HOME:?set CODEX_HOME to an authenticated, persistent Codex home}"
mkdir -p "$CODEX_HOME" "$CODEX_HOME/npm"

if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
  node_version="${GEAK_NODE_VERSION:-20.18.1}"
  case "$(uname -m)" in
    x86_64|amd64) node_arch=x64 ;;
    aarch64|arm64) node_arch=arm64 ;;
    *) die "unsupported architecture for portable Node.js install: $(uname -m)" ;;
  esac
  command -v curl >/dev/null 2>&1 || die "curl is required for the portable Node.js install"
  log "installing portable Node.js v$node_version ($node_arch)"
  mkdir -p "$CODEX_HOME/node"
  curl -fsSL --retry 4 \
    "https://nodejs.org/dist/v$node_version/node-v$node_version-linux-$node_arch.tar.xz" |
    tar -xJ -C "$CODEX_HOME/node" --strip-components=1
  export PATH="$CODEX_HOME/node/bin:$PATH"
fi

if ! command -v codex >/dev/null 2>&1; then
  log "installing Codex CLI under $CODEX_HOME/npm"
  npm install --quiet --no-audit --no-fund --prefix "$CODEX_HOME/npm" @openai/codex
  export PATH="$CODEX_HOME/npm/node_modules/.bin:$PATH"
fi
command -v codex >/dev/null 2>&1 || die "Codex CLI install completed but codex is not on PATH"

export GEAK_CODEX_BIN="$(command -v codex)"
log "version: $(codex --version)"
status="$(codex login status 2>&1 || true)"
printf '%s\n' "$status"
printf '%s' "$status" | grep -qi 'logged in' || die "Codex is not authenticated; run 'codex login' on the host and mount that CODEX_HOME"

model="${GEAK_CODEX_MODEL:-gpt-5.6-sol}"
effort="${GEAK_CODEX_REASONING_EFFORT:-high}"
log "probing $model (reasoning=$effort)"
codex exec --ephemeral --skip-git-repo-check --sandbox read-only \
  --model "$model" -c "model_reasoning_effort=\"$effort\"" \
  "Reply with exactly: SETUP OK" </dev/null >/dev/null
log "probe OK"
