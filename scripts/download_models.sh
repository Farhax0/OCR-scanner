#!/usr/bin/env bash
# Download Tesseract's most accurate ("tessdata_best") LSTM models into ./models.
# Usage: scripts/download_models.sh [lang ...]   (default: eng)
# Examples: scripts/download_models.sh eng hin ara fra deu
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models
langs=("${@:-eng}")
for l in "${langs[@]}" osd; do
  [ "$l" = osd ] && repo=tessdata || repo=tessdata_best
  echo "Downloading $l ($repo)..."
  curl -fsSL -o "models/$l.traineddata" \
    "https://raw.githubusercontent.com/tesseract-ocr/$repo/main/$l.traineddata"
done
echo "Done. Models in ./models are used automatically."
