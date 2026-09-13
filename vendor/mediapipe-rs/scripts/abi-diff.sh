#!/usr/bin/env bash
# Compare the public MediaPipe C headers between two refs and separate harmless
# renames from real layout changes.
#
# Run this at every version bump. Renames cost us type aliases; a field added,
# removed or reordered costs us a new entry in src/sys/compat.rs, and forgetting
# one does not fail loudly — the C side just reads fields from the wrong offset.
#
# Usage: scripts/abi-diff.sh v0.10.35 HEAD [path-to-mediapipe-checkout]
set -eu

OLD=${1:?usage: abi-diff.sh <old-ref> <new-ref> [mediapipe-checkout]}
NEW=${2:?usage: abi-diff.sh <old-ref> <new-ref> [mediapipe-checkout]}
REPO=${3:-../mediapipe}

cd "$(dirname "$0")/.."
[ -d "$REPO/.git" ] || { echo "no mediapipe checkout at $REPO" >&2; exit 1; }

strip() { grep -vE '^\s*//|^\s*$'; }

status=0
for header in $(cd vendor/headers && find mediapipe/tasks/c -name '*.h' | sort); do
  old=$(git -C "$REPO" show "$OLD:$header" 2>/dev/null || true)
  new=$(git -C "$REPO" show "$NEW:$header" 2>/dev/null || true)

  if [ -z "$old" ]; then echo "new file     $header"; continue; fi
  if [ -z "$new" ]; then echo "REMOVED      $header"; status=1; continue; fi

  d=$(diff <(printf '%s\n' "$old" | strip) <(printf '%s\n' "$new" | strip) || true)
  [ -n "$d" ] || continue

  # A pure rename changes identifiers but not the number of declarations, so
  # equal +/- counts is a necessary (not sufficient) condition. Re-wrapping a
  # long signature also trips this, hence "needs review" rather than a verdict.
  removed=$(printf '%s\n' "$d" | grep -c '^<' || true)
  added=$(printf '%s\n' "$d" | grep -c '^>' || true)
  if [ "$removed" -eq "$added" ]; then
    echo "rename-only? $header  ($added lines)"
  else
    echo "REVIEW       $header  (-$removed +$added)  <-- may need src/sys/compat.rs"
    printf '%s\n' "$d" | sed 's/^/    /'
    status=1
  fi
done

if [ "$status" -eq 0 ]; then
  echo "no candidate layout changes between $OLD and $NEW"
else
  echo
  echo "Review the entries above. Only field additions/removals/reordering need a"
  echo "compat entry; renames and reformatting do not."
fi
exit "$status"
