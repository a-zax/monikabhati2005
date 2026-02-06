#!/bin/bash
# Script to download IU-Xray dataset using Kaggle API

# Check for kaggle.json
if [ ! -f ~/.kaggle/kaggle.json ] && [ ! -f $USERPROFILE/.kaggle/kaggle.json ]; then
    echo "Error: kaggle.json not found in ~/.kaggle/ or %USERPROFILE%/.kaggle/"
    echo "Please download your API token from Kaggle and place it there."
    exit 1
fi

echo "Downloading IU-Xray dataset..."
mkdir -p data/raw
cd data/raw

# Download
kaggle datasets download -d raddar/chest-xrays-indiana-university --unzip -p iu_xray

echo "Download complete in data/raw/iu_xray"
