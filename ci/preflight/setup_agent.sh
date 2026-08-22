#!/usr/bin/env bash
# Install/probe the selected GEAK agent harness inside the runtime container.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "${GEAK_AGENT_BACKEND:-claude}" in
  codex)  exec bash "$HERE/setup_codex.sh" ;;
  claude) exec bash "$HERE/setup_claude.sh" ;;
  *) echo "ERROR: GEAK_AGENT_BACKEND must be codex or claude" >&2; exit 2 ;;
esac
