#!/bin/sh
# Downloads the models and test image used by tests/ and examples/ into models/.
# These are not vendored: they are Google-hosted assets, and the MediaPipe
# checkout does not ship them either (bazel fetches them on demand).
set -eu
cd "$(dirname "$0")/.."
mkdir -p models

fetch() {
  [ -f "models/$2" ] && { echo "have $2"; return; }
  echo "fetching $2"
  curl -fsSL -o "models/$2" "$1"
}

fetch https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite \
      blaze_face_short_range.tflite
fetch https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task \
      face_landmarker.task
fetch https://storage.googleapis.com/mediapipe-assets/portrait.jpg \
      portrait.jpg

ls -l models
