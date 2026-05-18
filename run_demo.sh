#!/bin/bash
set -e

echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate vid3d

echo "Running Video to Semantic 3D Pipeline on demo3.mp4..."
python src/main.py demo3.mp4 --workspace demo_output --fps 5.0

echo "Done! You can find the output in the demo_output directory."
