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
    bounding_boxes = []
    colors = np.random.rand(100, 3)
    
    if bbox_path.exists():
        print("Loading Semantic Bounding Boxes...")
        with open(bbox_path, 'r') as f:
            bounding_boxes = json.load(f)
            
        np.random.seed(42)
        colors = np.random.rand(100, 3)
        
    print("\nDetected Objects:")
    for i, bbox_data in enumerate(bounding_boxes):
        cls_id = bbox_data["class_id"]
        cls_name = bbox_data["class_name"]
        center = bbox_data["center"]
        R = np.array(bbox_data["R"])
        extent = bbox_data["extent"]
        
        color = colors[cls_id % 100]
        
        obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
        obb.color = color
        geometries.append(obb)
        
        # Add 3D Text Label using the modern tensor API converted to legacy
        try:
            t_mesh = o3d.t.geometry.TriangleMesh.create_text(cls_name, depth=0.05)
            text_mesh = t_mesh.to_legacy()
            text_mesh.compute_vertex_normals()
            text_mesh.paint_uniform_color(color)
            
            # Scale and position the text
            text_mesh.scale(0.15, center=text_mesh.get_center())
            label_pos = np.array(center) + np.array([0, extent[1]/2 + 0.1, 0])
            text_mesh.translate(label_pos, relative=False)
            
            geometries.append(text_mesh)
        except Exception as e:
            pass
        
        print(f" - [{cls_name}] (Color: R={color[0]:.2f}, G={color[1]:.2f}, B={color[2]:.2f})")

    import cv2
    
    print("\n=== Initializing Visualizer ===")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Semantic Bounding Boxes", width=1280, height=720)
    
    for g in geometries:
        vis.add_geometry(g)
        
    # Reset camera to fit geometries and render a few frames to populate the buffer
    vis.reset_camera_to_default()
    for _ in range(50):
        vis.poll_events()
        vis.update_renderer()
    
    # Capture the screen buffer
    img = vis.capture_screen_float_buffer(False)
    img = (np.asarray(img) * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Get camera parameters for projection
    ctr = vis.get_view_control()
    cam_params = ctr.convert_to_pinhole_camera_parameters()
    extrinsic = cam_params.extrinsic
    intrinsic = cam_params.intrinsic.intrinsic_matrix
    
    # Draw labels on the PNG
    print("Projecting labels onto automatic screenshot...")
    for bbox_data in bounding_boxes:
        cls_name = bbox_data["class_name"]
        center = np.array(bbox_data["center"])
        
        # World to Camera
        p_cam = extrinsic[:3, :3] @ center + extrinsic[:3, 3]
        if p_cam[2] > 0: # Only if in front of camera
            # Camera to Pixel
            p_pix = intrinsic @ p_cam
            u, v = int(p_pix[0] / p_pix[2]), int(p_pix[1] / p_pix[2])
            
            if 0 <= u < 1280 and 0 <= v < 720:
                # Draw a small dot and the text
                cv2.circle(img, (u, v), 5, (0, 0, 255), -1)
                cv2.putText(img, cls_name, (u + 10, v), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
    # Save automatically
    out_path = workspace / "view_general.png"
    cv2.imwrite(str(out_path), img)
    print(f"Saved automatic labeled screenshot to {out_path}")
    
    print("\n=== Interactive Controls ===")
    print("  - Left Click + Drag : Rotate")
    print("  - Right Click + Drag: Translate")
    print("  - Scroll Wheel      : Zoom")
    print("  - [Q]               : Close")
    
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=str, default="workspace", help="Path to workspace directory")
    args = parser.parse_args()
    visualize(args.workspace)
