#!/bin/bash

set -euo pipefail

FOLDER_URL="https://drive.google.com/drive/folders/1rs6s4e_eyTJCuE9bMdAJuAh1Isis2PfA?usp=sharing"
TARGET_DIR="datasets"
TMP_DIR=".tmp_gdrive_download"

echo "Preparing local datasets directory..."
mkdir -p "$TARGET_DIR"

# Install gdown if missing
if ! command -v gdown >/dev/null 2>&1; then
    echo "gdown not found. Installing..."
    python3 -m pip install gdown
fi

# Clean temporary download area
rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"

echo "Downloading Google Drive folder into temporary directory..."
gdown --folder "$FOLDER_URL" -O "$TMP_DIR"

echo "Copying only missing files into '$TARGET_DIR'..."

if command -v rsync >/dev/null 2>&1; then
    rsync -av --ignore-existing "$TMP_DIR"/ "$TARGET_DIR"/
else
    echo "rsync not found. Falling back to cp -rn ..."
    cp -rn "$TMP_DIR"/. "$TARGET_DIR"/
fi

echo "Cleaning up temporary files..."
rm -rf "$TMP_DIR"

echo "Done. Missing files were added to '$TARGET_DIR'. Existing files were left unchanged."