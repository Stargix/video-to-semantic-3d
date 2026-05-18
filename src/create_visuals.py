import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import imageio
import open3d as o3d
import argparse

def create_summary_graphic(workspace_dir="workspace"):
    workspace = Path(workspace_dir)
    images_dir = workspace / "images"
    depths_dir = workspace / "depths"
    semantics_dir = workspace / "semantics"
    
    img_paths = sorted(glob.glob(str(images_dir / "*.jpg")))
    
    if not img_paths:
        print("No images found to create graphic.")
        return
        
    # Select 3 evenly spaced frames
    indices = np.linspace(0, len(img_paths)-1, 3, dtype=int)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    fig.suptitle("Video to Semantic 3D: Processing Steps", fontsize=16)
    
    for i, idx in enumerate(indices):
        img_path = img_paths[idx]
        stem = Path(img_path).stem
        
        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load depth
        depth_path = depths_dir / f"{stem}.npy"
        depth = np.load(depth_path) if depth_path.exists() else np.zeros_like(img[:,:,0])
        
        # Load semantics
        sem_path = semantics_dir / f"{stem}.npy"
        sem = np.load(sem_path) if sem_path.exists() else np.zeros_like(img[:,:,0])
        
        axes[i, 0].imshow(img)
        if i == 0: axes[i, 0].set_title("Original Frame")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(depth, cmap='magma')
        if i == 0: axes[i, 1].set_title("Dense Depth (DepthAnythingV2)")
        axes[i, 1].axis('off')
        
        # Visualize semantics uniquely
        colored_sem = plt.get_cmap('tab20')(sem % 20)
        colored_sem[sem == -1] = [0,0,0,1] # Black for background
        axes[i, 2].imshow(colored_sem)
        if i == 0: axes[i, 2].set_title("Semantics (YOLOv8-seg)")
        axes[i, 2].axis('off')
        
    plt.tight_layout()
    out_path = workspace / "pipeline_summary.png"
    plt.savefig(str(out_path), dpi=150)
    print(f"Saved pipeline summary graphic to {out_path}")


def create_pointcloud_gif(workspace_dir="workspace"):
    workspace = Path(workspace_dir)
    mesh_path = workspace / "scene_mesh.ply"
    rgb_path = workspace / "scene_rgb.ply"
    
    if mesh_path.exists():
        pcd = o3d.io.read_triangle_mesh(str(mesh_path))
    elif rgb_path.exists():
        pcd = o3d.io.read_point_cloud(str(rgb_path))
    else:
        print("Point cloud / Mesh not found. Skipping GIF generation.")
        return
    
    print("Generating 3D Point Cloud GIF (this will open a window briefly)...")
    
    vis = o3d.visualization.Visualizer()
    success = vis.create_window(visible=False) # Try hidden window
    if not success:
        print("Warning: Could not create OpenGL window for GIF. Skipping GIF generation.")
        vis.destroy_window()
        return
        
    vis.add_geometry(pcd)
    
    opt = vis.get_render_option()
    if opt is None:
        print("Warning: Could not get render options for GIF. Skipping GIF generation.")
        vis.destroy_window()
        return
        
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 2.0
    
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    
    frames = []
    # Rotate slightly
    for i in range(30):
        ctr.rotate(10.0, 0.0) # rotate 10 degrees along Y axis
        vis.poll_events()
        vis.update_renderer()
        
        img = vis.capture_screen_float_buffer(False)
        img_np = (np.asarray(img) * 255).astype(np.uint8)
        frames.append(img_np)
        
    vis.destroy_window()
    
    gif_path = workspace / "3d_reconstruction.gif"
    imageio.mimsave(str(gif_path), frames, fps=10)
    print(f"Saved rotating point cloud GIF to {gif_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=str, default="workspace")
    args = parser.parse_args()
    
    create_summary_graphic(args.workspace)
    try:
        create_pointcloud_gif(args.workspace)
    except Exception as e:
        print(f"Could not generate 3D GIF (likely due to missing display environment): {e}")
