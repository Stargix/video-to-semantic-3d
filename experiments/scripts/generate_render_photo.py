import json
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d

def generate_render_photo(workspace_dir: str):
    workspace = Path(workspace_dir)
    mesh_path = workspace / "scene_mesh.ply"
    rgb_path = workspace / "scene_rgb.ply"
    bbox_path = workspace / "bounding_boxes.json"
    out_path = workspace / "view_general.png"
    
    # 1. Check if we have 3D data to render
    if mesh_path.exists():
        print(f"Loading mesh from {mesh_path.name}...")
        geom = o3d.io.read_triangle_mesh(str(mesh_path))
        geom.compute_vertex_normals()
        is_mesh = True
    elif rgb_path.exists():
        print(f"Loading point cloud from {rgb_path.name}...")
        geom = o3d.io.read_point_cloud(str(rgb_path))
        is_mesh = False
    else:
        print("No 3D mesh or point cloud found. Skipping render generation.")
        return False

    # 2. Setup offscreen renderer
    width, height = 1280, 720
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    scene = renderer.scene
    scene.set_background([1.0, 1.0, 1.0, 1.0]) # white background

    # 3. Add scene geometry
    material = o3d.visualization.rendering.MaterialRecord()
    if is_mesh:
        material.shader = "defaultLit"
    else:
        material.shader = "defaultLit"
        material.point_size = 3.0
    scene.add_geometry("scene_geom", geom, material)

    # 4. Add semantic bounding boxes as 3D wireframe boxes
    bounding_boxes = []
    if bbox_path.exists():
        try:
            with open(bbox_path, 'r') as f:
                bounding_boxes = json.load(f)
        except Exception as e:
            print(f"Error loading bounding boxes: {e}")

    np.random.seed(42)
    colors = np.random.rand(100, 3)

    for i, bbox_data in enumerate(bounding_boxes):
        center = np.array(bbox_data["center"])
        R = np.array(bbox_data["R"])
        extent = np.array(bbox_data["extent"])
        cls_name = bbox_data.get("class_name", "object")
        cls_id = bbox_data.get("class_id", 0)
        
        color = colors[cls_id % 100]
        
        # Create OBB and convert to LineSet for beautiful wireframe
        obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
        line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
        line_set.paint_uniform_color(color)
        
        material_lines = o3d.visualization.rendering.MaterialRecord()
        material_lines.shader = "unlitLine"
        material_lines.line_width = 4.0
        
        scene.add_geometry(f"box_{i}_{cls_name}", line_set, material_lines)

    # 5. Position camera dynamically based on scene bounds
    bbox = geom.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()
    
    # Calculate optimal camera distance and placement
    distance = np.max(extent) * 1.2
    eye = center + np.array([-distance * 0.7, -distance * 0.7, distance * 0.7])
    up = np.array([0.0, 0.0, 1.0]) # Z-up standard
    
    # Set perspective projection with correct aspect ratio to prevent squashing
    aspect_ratio = width / height
    scene.camera.set_projection(45.0, aspect_ratio, 0.1, 1000.0, o3d.visualization.rendering.Camera.FovType.Vertical)
    scene.camera.look_at(center, eye, up)
    
    # Add ambient and directional light for beautiful illumination
    scene.scene.set_indirect_light_intensity(2.0)
    scene.scene.enable_indirect_light(True)

    # 6. Render and save image
    print("Rendering 3D scene headlessly...")
    try:
        image = renderer.render_to_image()
        o3d.io.write_image(str(out_path), image)
        print(f"[✓] Headless render saved successfully: {out_path.name}")
        return True
    except Exception as e:
        print(f"[!] Headless render failed: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=str, required=True, help="Path to workspace directory")
    args = parser.parse_args()
    generate_render_photo(args.workspace)
