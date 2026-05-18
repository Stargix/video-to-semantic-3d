#!/bin/bash
set -e

echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate vid3d

echo "Running Video to Semantic 3D Pipeline on demo3.mp4..."
OUTPUT_DIR="demo_output_$(date +%Y%m%d_%H%M%S)"
python src/main.py demo3.mp4 --workspace $OUTPUT_DIR --fps 5.0

echo "Done! You can find the output in the $OUTPUT_DIR directory."
echo "To visualize, run: python src/visualize.py --workspace $OUTPUT_DIR"
