import os
import cv2
import numpy as np
import open3d as o3d
import pycolmap
from pathlib import Path
from tqdm import tqdm
from scipy.optimize import least_squares

class FusionEngine:
    def __init__(self, workspace_dir: str = "workspace"):
        self.workspace_dir = Path(workspace_dir)
        self.images_dir = self.workspace_dir / "images"
        self.depth_dir = self.workspace_dir / "depths"
        self.masks_dir = self.workspace_dir / "semantics"
        self.sparse_dir = self.workspace_dir / "sparse"
        self.output_path = self.workspace_dir / "semantic_scene.ply"
        
        # We will load the COCO class names to use for coloring or metadata
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt") # Just to get names
        self.class_names = model.names
        
    def _align_depth(self, rel_depth, points_3d, camera, image_name, points_2d, point3D_ids):
        """
        Aligns the relative depth to the sparse metric 3D points from COLMAP.
        """
        # Get ground truth metric depths for the sparse points
        metric_depths = []
        rel_depths_for_points = []
        
        # Camera extrinsics (world to camera)
        pose = camera.cam_from_world()
        R = pose.rotation.matrix()
        t = pose.translation
        
        for i, p3D_id in enumerate(point3D_ids):
            if p3D_id == -1: continue # Not a valid 3D point
            if p3D_id not in points_3d: continue
            
            p3D = points_3d[p3D_id].xyz
            
            # Project to camera space to get metric depth
            p_cam = R @ p3D + t
            z_metric = p_cam[2]
            
            if z_metric <= 0: continue
            
            # Get corresponding 2D point (from COLMAP feature extraction)
            x, y = points_2d[i].xy
            x, y = int(round(x)), int(round(y))
            
            if y >= rel_depth.shape[0] or x >= rel_depth.shape[1] or y < 0 or x < 0:
                continue
                
            z_rel = rel_depth[y, x]
            
            metric_depths.append(z_metric)
            rel_depths_for_points.append(z_rel)
            
        if len(metric_depths) < 10:
            return None # Not enough points for reliable alignment
            
        metric_depths = np.array(metric_depths)
        rel_depths_for_points = np.array(rel_depths_for_points)
        
        # We want to find s, t such that metric_depth = s * rel_depth + t
        # Using least squares
        def residuals(vars, d_rel, d_metric):
            s, t = vars
            return (s * d_rel + t) - d_metric
            
        # Initial guess
        s_guess = np.median(metric_depths) / (np.median(rel_depths_for_points) + 1e-6)
        res = least_squares(residuals, x0=[s_guess, 0.0], args=(rel_depths_for_points, metric_depths), loss='soft_l1')
        
        s, t = res.x
        
        # Apply alignment
        aligned_depth = s * rel_depth + t
        aligned_depth[aligned_depth < 0] = 0
        return aligned_depth
        
    def _unproject(self, aligned_depth, image, mask, camera):
        """
        Unprojects a depth map into a 3D point cloud.
        """
        # Camera intrinsics
        h, w = aligned_depth.shape
        # pycolmap 4.0 API for camera parameters:
        # camera.camera is a pycolmap.Camera object
        # we can use camera.camera.focal_length, camera.camera.principal_point etc
        # or we can use calibration matrix if provided.
        # camera.camera.calibration_matrix()
        
        # params = camera.camera.params
        
        # Let's dynamically fetch them to avoid param index issues
        # Actually camera.camera.params is a property containing focal length and principal points
        K = camera.camera.calibration_matrix()
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
            
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        z = aligned_depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        # Camera to World
        pts_cam = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        
        pose = camera.cam_from_world()
        R = pose.rotation.matrix()
        t = pose.translation
        
        # R * p_w + t = p_c  => p_w = R^T * (p_c - t)
        pts_world = (pts_cam - t) @ R
        
        colors = image.reshape(-1, 3) / 255.0
        semantics = mask.reshape(-1)
        
        # Filter invalid depths
        valid = (z.reshape(-1) > 0) & (z.reshape(-1) < 100) # Filter very far points
        
        return pts_world[valid], colors[valid], semantics[valid]
        
    def run_fusion(self):
        print("Starting 3D Fusion...")
        
        if not self.sparse_dir.exists():
            raise FileNotFoundError("Sparse reconstruction not found. Run SfM first.")
            
        reconstruction = pycolmap.Reconstruction(self.sparse_dir)
        points_3d = reconstruction.points3D
        
        all_pts = []
        all_colors = []
        all_semantics = []
        
        print("Aligning depths and unprojecting...")
        for image_id, camera in tqdm(reconstruction.images.items()):
            img_name = camera.name
            img_stem = Path(img_name).stem
            
            img_path = self.images_dir / img_name
            depth_path = self.depth_dir / f"{img_stem}.npy"
            mask_path = self.masks_dir / f"{img_stem}.npy"
            
            if not img_path.exists() or not depth_path.exists() or not mask_path.exists():
                continue
                
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            rel_depth = np.load(depth_path)
            mask = np.load(mask_path)
            
            # Get 2D points and 3D point IDs for this image
            points_2d = camera.points2D
            point3D_ids = [p.point3D_id for p in points_2d]
            
            # Align
            aligned_depth = self._align_depth(rel_depth, points_3d, camera, img_name, points_2d, point3D_ids)
            if aligned_depth is None:
                continue
                
            # Unproject
            pts, colors, semantics = self._unproject(aligned_depth, image, mask, camera)
            
            # Subsample to avoid blowing up memory (e.g. keep 10%)
            subsample = np.random.choice(len(pts), size=int(len(pts)*0.05), replace=False)
            
            all_pts.append(pts[subsample])
            all_colors.append(colors[subsample])
            all_semantics.append(semantics[subsample])
            
        print("Fusing point clouds...")
        pts_stacked = np.vstack(all_pts)
        colors_stacked = np.vstack(all_colors)
        semantics_stacked = np.hstack(all_semantics)
        
        # Create Open3D PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_stacked)
        pcd.colors = o3d.utility.Vector3dVector(colors_stacked)
        
        # We can't directly save semantics as an array in a standard ply easily using o3d without custom headers,
        # but we can map semantic IDs to a custom color palette and save that as well.
        # Let's save a custom numpy structure, or encode it as normals for visualization.
        # Actually, let's just create a dictionary of colors for semantic IDs.
        np.random.seed(42)
        semantic_colors = np.random.rand(100, 3) 
        semantic_colors[0] = [0.8, 0.8, 0.8] # -1 + 1 = 0 (Background)
        
        sem_color_mapped = semantic_colors[semantics_stacked + 1]
        
        pcd_semantic = o3d.geometry.PointCloud()
        pcd_semantic.points = o3d.utility.Vector3dVector(pts_stacked)
        pcd_semantic.colors = o3d.utility.Vector3dVector(sem_color_mapped)
        
        print("Downsampling and cleaning point cloud...")
        # Voxel downsample
        pcd = pcd.voxel_down_sample(0.02)
        # Apply same downsampling to semantic pcd (it will yield exact same points/order)
        pcd_semantic = pcd_semantic.voxel_down_sample(0.02)
        
        # Statistical outlier removal
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd_semantic = pcd_semantic.select_by_index(ind)
        
        o3d.io.write_point_cloud(str(self.workspace_dir / "scene_rgb.ply"), pcd)
        o3d.io.write_point_cloud(str(self.workspace_dir / "scene_semantic.ply"), pcd_semantic)
        print(f"Fusion complete. Saved to {self.workspace_dir}")

if __name__ == "__main__":
    # fusion = FusionEngine()
    # fusion.run_fusion()
    pass
