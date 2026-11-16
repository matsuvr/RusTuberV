#!/bin/bash
# Build the upstream pinned framework and place it beside a standalone binary,
# or into an app's Contents/Frameworks (the caller supplies the destination).
set -euo pipefail
cd "$(dirname "$0")/.."
test "$(uname -s)" = Darwin
revision=f4761677a45b8034a3c2069ec0f3d2553da81fba
destination="${1:-target/release}"
mkdir -p "$destination"
destination="$(cd "$destination" && pwd)"
work=$(mktemp -d "${TMPDIR:-/tmp}/rustuberv-syphon.XXXXXX")
trap 'rm -rf "$work"' EXIT
git -C "$work" init --quiet
git -C "$work" remote add origin https://github.com/syphon/syphon-framework.git
git -C "$work" fetch --depth 1 origin "$revision"
git -C "$work" checkout --detach FETCH_HEAD
test "$(git -C "$work" rev-parse HEAD)" = "$revision"
xcodebuild -project "$work/Syphon.xcodeproj" -scheme Syphon \
    -configuration Release -derivedDataPath "$work/build" \
    ARCHS="$(uname -m)" ONLY_ACTIVE_ARCH=YES MACOSX_DEPLOYMENT_TARGET=14.0 \
    CODE_SIGNING_ALLOWED=NO build
framework="$work/build/Build/Products/Release/Syphon.framework"
test -f "$framework/Syphon"
ditto "$framework" "$destination/Syphon.framework"
codesign --force --sign - "$destination/Syphon.framework"
codesign --verify --strict "$destination/Syphon.framework"
