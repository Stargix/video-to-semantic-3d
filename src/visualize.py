import os
os.environ["GDK_BACKEND"] = "x11"
os.environ["XDG_SESSION_TYPE"] = "x11"

import open3d as o3d
import argparse
from pathlib import Path
import numpy as np
import json

def visualize(workspace_dir="workspace"):
    workspace = Path(workspace_dir)
    mesh_path = workspace / "scene_mesh.ply"
    rgb_path = workspace / "scene_rgb.ply"
    bbox_path = workspace / "bounding_boxes.json"
    
    geometries = []
    
    if mesh_path.exists():
        print("Loading TSDF Triangle Mesh...")
        scene_geom = o3d.io.read_triangle_mesh(str(mesh_path))
        geometries.append(scene_geom)
    elif rgb_path.exists():
        print("Loading RGB point cloud...")
        scene_geom = o3d.io.read_point_cloud(str(rgb_path))
        geometries.append(scene_geom)
    else:
        print(f"Neither scene_mesh.ply nor scene_rgb.ply found in {workspace}. Please run the pipeline first.")
        return
    
    if bbox_path.exists():
        print("Loading Semantic Bounding Boxes...")
        with open(bbox_path, 'r') as f:
            bounding_boxes = json.load(f)
            
        np.random.seed(42)
        # Create a consistent color mapping for up to 100 classes
        colors = np.random.rand(100, 3)
        
        print("\nDetected Objects:")
        for bbox_data in bounding_boxes:
            cls_id = bbox_data["class_id"]
            cls_name = bbox_data["class_name"]
            center = bbox_data["center"]
            R = np.array(bbox_data["R"])
            extent = bbox_data["extent"]
            
            color = colors[cls_id % 100]
            
            obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
            obb.color = color
            
            geometries.append(obb)
            print(f" - [{cls_name}] (Color: R={color[0]:.2f}, G={color[1]:.2f}, B={color[2]:.2f})")
    else:
        print("No bounding_boxes.json found. Showing only RGB cloud.")
        
    print("\n=== Interactive Visualizer ===")
    print("Controls:")
    print("  - Left Click + Drag : Rotate")
    print("  - Right Click + Drag: Translate")
    print("  - Scroll Wheel      : Zoom")
    print("  - [H]               : Reset View")
    print("  - [Q]               : Close")
    
    # Add a coordinate frame for reference
    bbox = scene_geom.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max(extent)*0.1, origin=[0, 0, 0])
    geometries.append(coord_frame)
    
    o3d.visualization.draw_geometries(geometries, 
                                      window_name="3D Semantic Bounding Boxes",
                                      width=1280, height=720)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=str, default="workspace", help="Path to workspace directory")
    args = parser.parse_args()
    visualize(args.workspace)
