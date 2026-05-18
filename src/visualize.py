import os
os.environ["GDK_BACKEND"] = "x11"
os.environ["XDG_SESSION_TYPE"] = "x11"

import open3d as o3d
import argparse
from pathlib import Path
import numpy as np

def visualize(workspace_dir="workspace"):
    workspace = Path(workspace_dir)
    rgb_path = workspace / "scene_rgb.ply"
    semantic_path = workspace / "scene_semantic.ply"
    
    if not rgb_path.exists() or not semantic_path.exists():
        print(f"Point clouds not found in {workspace}. Please run the pipeline first.")
        return
        
    print("Loading point clouds...")
    pcd_rgb = o3d.io.read_point_cloud(str(rgb_path))
    pcd_semantic = o3d.io.read_point_cloud(str(semantic_path))
    
    # Calculate offset to place them side by side
    bbox = pcd_rgb.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    offset = extent[0] * 1.5 # Offset along X axis
    
    # Shift the semantic cloud
    pcd_semantic.translate([offset, 0, 0])
    
    print("\n=== Interactive Visualizer ===")
    print("Controls:")
    print("  - Left Click + Drag : Rotate")
    print("  - Right Click + Drag: Translate")
    print("  - Scroll Wheel      : Zoom")
    print("  - [H]               : Reset View")
    print("  - [Q]               : Close")
    print("\nShowing RGB (Left) and Semantic (Right) side-by-side.")
    
    # Add a coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=extent[2]*0.2, origin=[0, 0, 0])
    
    o3d.visualization.draw_geometries([pcd_rgb, pcd_semantic, coord_frame], 
                                      window_name="3D Reconstruction: RGB vs Semantic",
                                      width=1280, height=720)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=str, default="workspace", help="Path to workspace directory")
    args = parser.parse_args()
    visualize(args.workspace)
