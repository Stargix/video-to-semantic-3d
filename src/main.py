import argparse
import os
from pathlib import Path

from video_processor import VideoProcessor
from sfm_mapper import SfMMapper
from depth_estimator import DepthEstimator
from semantic_segmenter import SemanticSegmenter
from fusion_engine import FusionEngine

def main():
    parser = argparse.ArgumentParser(description="Video to Semantic 3D Reconstruction")
    parser.add_argument("video_path", type=str, help="Path to input video")
    parser.add_argument("--workspace", type=str, default="workspace", help="Directory to save outputs")
    parser.add_argument("--fps", type=float, default=5.0, help="Frames per second to extract")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file {args.video_path} not found.")
        return
        
    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    
    print("=== Step 1: Extracting Frames ===")
    vp = VideoProcessor(args.workspace)
    vp.extract_frames(args.video_path, fps=args.fps)
    
    print("\n=== Step 2: Sparse Structure from Motion (COLMAP) ===")
    sfm = SfMMapper(args.workspace)
    sfm.run_reconstruction()
    
    print("\n=== Step 3: Dense Depth Estimation ===")
    depth_est = DepthEstimator(args.workspace)
    depth_est.estimate_depths()
    
    print("\n=== Step 4: Semantic Segmentation ===")
    seg = SemanticSegmenter(args.workspace)
    seg.extract_semantics()
    
    print("\n=== Step 5: Metric Alignment & 3D Fusion ===")
    fusion = FusionEngine(args.workspace)
    fusion.run_fusion()
    
    print("\n=== Step 6: Generating Visuals (Graphs and GIFs) ===")
    try:
        from create_visuals import create_summary_graphic, create_pointcloud_gif
        create_summary_graphic(args.workspace)
        create_pointcloud_gif(args.workspace)
    except Exception as e:
        print(f"Warning: Could not generate visuals automatically: {e}")
    
    print("\n=== Pipeline Complete ===")
    print(f"Reconstructed point clouds and visuals saved to {workspace}")
    print("Run `python src/visualize.py --workspace workspace` to view interactive 3D results.")

if __name__ == "__main__":
    main()
