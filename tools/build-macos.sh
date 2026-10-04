#!/bin/bash
# Local macOS app using the installed NDI SDK (not a redistributable package).
set -euo pipefail
cd "$(dirname "$0")/.."

cargo build --locked --release -p vtuber-desktop

app="target/release/RusTuberV.app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources/assets/models"
cp target/release/RusTuberV "$app/Contents/MacOS/RusTuberV"
cp apps/desktop/macos/Info.plist "$app/Contents/Info.plist"

# Reuse the supplied PNG sizes, resizing only the missing 512px image.
iconset="target/release/RusTuberV.iconset"
mkdir -p "$iconset"
for size in 16 32 128 256; do
    cp "assets/icons/rustuberv-app-icon-$size.png" "$iconset/icon_${size}x${size}.png"
done
for size in 16 32 128 512; do
    cp "assets/icons/rustuberv-app-icon-$((size * 2)).png" "$iconset/icon_${size}x${size}@2x.png"
done
sips --resampleHeightWidth 512 512 assets/icons/rustuberv-app-icon-1024.png \
    --out "$iconset/icon_512x512.png" >/dev/null
cp "$iconset/icon_512x512.png" "$iconset/icon_256x256@2x.png"
iconutil --convert icns "$iconset" --output "$app/Contents/Resources/rustuberv-app.icns"

cp assets/models/manifest.toml "$app/Contents/Resources/assets/models/manifest.toml"
# Keep local tracking edits when updating an existing bundle.
if [ ! -f "$app/Contents/Resources/tracking_profile.toml" ]; then
    cp tracking_profile.toml "$app/Contents/Resources/tracking_profile.toml"
fi
codesign --force --sign - "$app"
# Launch Services uses the bundle's modification time to refresh its icon.
touch "$app"
printf 'Built %s/%s\n' "$PWD" "$app"
