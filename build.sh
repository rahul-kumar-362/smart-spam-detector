#!/bin/bash
set -e

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Retraining model..."
python train.py

echo "Build complete!"
