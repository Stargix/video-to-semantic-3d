#!/bin/bash
set -e

echo "Activating conda environment..."
source /home/stargix/miniconda3/etc/profile.d/conda.sh
conda activate vid3d

echo "Running Video to Semantic 3D Pipeline on inputs/demo3.mp4..."
OUTPUT_DIR="outputs/demo_output_$(date +%Y%m%d_%H%M%S)"
python src/main.py inputs/demo3.mp4 --workspace $OUTPUT_DIR --fps 5.0

echo "Done! You can find the output in the $OUTPUT_DIR directory."
echo "To visualize, run: python src/visualize.py --workspace $OUTPUT_DIR"
