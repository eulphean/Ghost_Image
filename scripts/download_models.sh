#!/usr/bin/env bash
# Download the MediaPipe person-segmentation models into models/.
# The files are git-ignored. Safe to re-run.
#
# selfie_segmenter.tflite is the default (config paths.model_file).
# selfie_multiclass_256x256.tflite labels hair, body, face, clothes and
# "others", which can hold onto fingers the single-class model drops. Point
# paths.model_file at it to compare the two on the Pi.

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p models

fetch() {
    local url="$1"
    local dest="$2"
    echo "==> $dest"
    curl --fail --location --silent --show-error --output "$dest" "$url"
}

fetch \
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite" \
    "models/selfie_segmenter.tflite"

fetch \
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite" \
    "models/selfie_multiclass_256x256.tflite"

echo "Done."
