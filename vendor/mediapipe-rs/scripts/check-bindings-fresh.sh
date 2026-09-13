#!/usr/bin/env bash
# Fails when the vendored headers are staged without regenerated bindings.
#
# This is the cheap half of the check: it needs no libclang, so it can run on
# every commit. CI does the expensive half — regenerate and diff — which also
# catches hand-edits to the generated file.
set -euo pipefail

changed=$(git diff --cached --name-only)

headers_changed=$(grep -cE '^vendor/(wrapper\.h|headers/)' <<<"$changed" || true)
bindings_changed=$(grep -cE '^src/sys/bindings\.rs$' <<<"$changed" || true)

if [ "$headers_changed" -gt 0 ] && [ "$bindings_changed" -eq 0 ]; then
  echo "vendored headers changed but src/sys/bindings.rs did not." >&2
  echo "Run ./scripts/gen-bindings.sh and stage the result." >&2
  exit 1
fi
