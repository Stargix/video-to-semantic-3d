import os
import cv2
import numpy as np
import open3d as o3d
import pycolmap
from pathlib import Path
from tqdm import tqdm
from scipy.optimize import least_squares
import json

class FusionEngine:
    def __init__(self, workspace_dir: str = "workspace"):
        self.workspace_dir = Path(workspace_dir)
        self.images_dir = self.workspace_dir / "images"
        self.depth_dir = self.workspace_dir / "depths"
        self.masks_dir = self.workspace_dir / "semantics"
        self.sparse_dir = self.workspace_dir / "sparse"
        self.output_path = self.workspace_dir / "scene_mesh.ply"
        
        # We will load the COCO class names to use for metadata
        try:
            from ultralytics import YOLO
            model = YOLO("yolov8n.pt") 
            self.class_names = model.names
        except:
            self.class_names = {i: f"Class_{i}" for i in range(100)}
        
    def _align_depth(self, rel_depth, points_3d, camera, image_name, points_2d, point3D_ids):
        metric_depths = []
        rel_depths_for_points = []
        
        pose = camera.cam_from_world()
        R = pose.rotation.matrix()
        t = pose.translation
        
        for i, p3D_id in enumerate(point3D_ids):
            if p3D_id == -1: continue 
            if p3D_id not in points_3d: continue
            
            p3D = points_3d[p3D_id].xyz
            p_cam = R @ p3D + t
            z_metric = p_cam[2]
            
            if z_metric <= 0: continue
            
            x, y = points_2d[i].xy
            x, y = int(round(x)), int(round(y))
            
            if y >= rel_depth.shape[0] or x >= rel_depth.shape[1] or y < 0 or x < 0:
                continue
                
            z_rel = rel_depth[y, x]
            metric_depths.append(z_metric)
            rel_depths_for_points.append(z_rel)
            
        if len(metric_depths) < 10:
            return None
            
        metric_depths = np.array(metric_depths)
        rel_depths_for_points = np.array(rel_depths_for_points)
        
        def residuals(vars, d_rel, d_metric):
            s, t = vars
            return (s * d_rel + t) - d_metric
            
        s_guess = np.median(metric_depths) / (np.median(rel_depths_for_points) + 1e-6)
        res = least_squares(residuals, x0=[s_guess, 0.0], args=(rel_depths_for_points, metric_depths), loss='soft_l1')
        
        s, t = res.x
        aligned_depth = s * rel_depth + t
        aligned_depth[aligned_depth < 0] = 0
        return aligned_depth
        
    def _unproject_semantics(self, aligned_depth, mask, camera):
        valid_mask = (mask != -1) & (aligned_depth > 0) & (aligned_depth < 100)
        if not np.any(valid_mask):
            return np.array([]), np.array([])
            
        K = camera.camera.calibration_matrix()
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        ys, xs = np.where(valid_mask)
        z = aligned_depth[valid_mask]
        
        x = (xs - cx) * z / fx
        y = (ys - cy) * z / fy
        
        pts_cam = np.stack((x, y, z), axis=-1)
        
        pose = camera.cam_from_world()
        R = pose.rotation.matrix()
        t = pose.translation
        
        pts_world = (pts_cam - t) @ R
        semantics = mask[valid_mask]
        
        return pts_world, semantics
        
    def run_fusion(self):
        print("Starting 3D Fusion (TSDF Volume Integration)...")
        
        if not self.sparse_dir.exists():
            raise FileNotFoundError("Sparse reconstruction not found. Run SfM first.")
            
        reconstruction = pycolmap.Reconstruction(self.sparse_dir)
        points_3d = reconstruction.points3D
        
        # Initialize TSDF Volume
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=0.08, # ~8cm voxels. Works well for indoor metric scale.
            sdf_trunc=0.3,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        
        all_sem_pts = []
        all_sem_ids = []
        
        print("Integrating TSDF and extracting semantics...")
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
            
            points_2d = camera.points2D
            point3D_ids = [p.point3D_id for p in points_2d]
            
            aligned_depth = self._align_depth(rel_depth, points_3d, camera, img_name, points_2d, point3D_ids)
            if aligned_depth is None: continue
            
            # 1. TSDF Integration
            h, w = aligned_depth.shape
            color_o3d = o3d.geometry.Image(image)
            depth_o3d = o3d.geometry.Image(aligned_depth.astype(np.float32))
            
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d, depth_scale=1.0, depth_trunc=100.0, convert_rgb_to_intensity=False
            )
            
            K = camera.camera.calibration_matrix()
            intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, K[0,0], K[1,1], K[0,2], K[1,2])
            
            pose = camera.cam_from_world()
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = pose.rotation.matrix()
            extrinsic[:3, 3] = pose.translation
            
            volume.integrate(rgbd, intrinsic, extrinsic)
            
            # 2. Semantic Extraction
            pts, semantics = self._unproject_semantics(aligned_depth, mask, camera)
            if len(pts) > 0:
                all_sem_pts.append(pts)
                all_sem_ids.append(semantics)
                
        print("Extracting and Cleaning TSDF Mesh...")
        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        
        # 1. Clean mesh: Remove only small floating noise clusters, keep all main components
        try:
            triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
            triangle_clusters = np.asarray(triangle_clusters)
            cluster_n_triangles = np.asarray(cluster_n_triangles)
            if len(cluster_n_triangles) > 0:
                # Remove clusters that are smaller than 5000 triangles (small floating noise)
                triangles_to_remove = cluster_n_triangles[triangle_clusters] < 5000
                mesh.remove_triangles_by_mask(triangles_to_remove)
                mesh.remove_unreferenced_vertices()
        except Exception as e:
            print(f"Warning: Could not clean mesh components: {e}")

        # 2. Coordinate System Alignment (OpenCV to OpenGL)
        # COLMAP is Y-down, Z-forward. Open3D visualizer expects Y-up, Z-backward.
        # We apply a 180-degree rotation around X-axis.
        flip_mat = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  0, 1]
        ])
        mesh.transform(flip_mat)
        
        o3d.io.write_triangle_mesh(str(self.workspace_dir / "scene_mesh.ply"), mesh)
        
        print("Extracting Semantic 3D Bounding Boxes...")
        if not all_sem_pts:
            print("No semantics found.")
            return
            
        pts_stacked = np.vstack(all_sem_pts)
        semantics_stacked = np.hstack(all_sem_ids)
        
        # Apply the same OpenCV -> OpenGL coordinate flip to semantic points
        pts_stacked_homo = np.hstack((pts_stacked, np.ones((pts_stacked.shape[0], 1))))
        pts_stacked = (flip_mat @ pts_stacked_homo.T).T[:, :3]
        
        bounding_boxes = []
        unique_classes = np.unique(semantics_stacked)
        
        all_extracted_boxes = []
        
        for cls_id in unique_classes:
            class_mask = (semantics_stacked == cls_id)
            class_pts = pts_stacked[class_mask]
            
            if len(class_pts) < 1000:
                continue
                
            temp_pcd = o3d.geometry.PointCloud()
            temp_pcd.points = o3d.utility.Vector3dVector(class_pts)
            temp_pcd = temp_pcd.voxel_down_sample(0.1)
            
            if len(temp_pcd.points) < 100:
                continue
            
            labels = np.array(temp_pcd.cluster_dbscan(eps=1.2, min_points=30, print_progress=False))
            if len(labels) == 0: continue
                
            max_label = labels.max()
            class_name = self.class_names[cls_id] if cls_id in self.class_names else f"Class_{cls_id}"
            
            for i in range(max_label + 1):
                cluster_mask = (labels == i)
                cluster_pts = np.asarray(temp_pcd.points)[cluster_mask]
                
                if len(cluster_pts) < 80: continue
                    
                cluster_pcd = o3d.geometry.PointCloud()
                cluster_pcd.points = o3d.utility.Vector3dVector(cluster_pts)
                
                try:
                    obb = cluster_pcd.get_oriented_bounding_box()
                    if obb.volume() < 0.5 or obb.volume() > 50000.0:
                        continue
                        
                    all_extracted_boxes.append({
                        "class_id": int(cls_id),
                        "class_name": class_name,
                        "center": obb.center.tolist(),
                        "R": obb.R.tolist(),
                        "extent": obb.extent.tolist(),
                        "volume": obb.volume()
                    })
                except Exception:
                    pass
            
        # Cross-Class Spatial NMS
        all_extracted_boxes.sort(key=lambda x: x["volume"], reverse=True)
        kept_boxes = []
        for box in all_extracted_boxes:
            c1 = np.array(box["center"])
            overlap = False
            for kept in kept_boxes:
                c2 = np.array(kept["center"])
                dist = np.linalg.norm(c1 - c2)
                max_ext = max(max(box["extent"]), max(kept["extent"]))
                
                # If they are different classes, we are aggressive to remove false positives (e.g. toilet inside couch)
                if box["class_id"] != kept["class_id"]:
                    if dist < max_ext * 0.7:
                        overlap = True
                        break
                # If they are the same class, we only remove if they are almost duplicates
                else:
                    if dist < max_ext * 0.3:
                        overlap = True
                        break
            
            if not overlap:
                kept_boxes.append(box)
                
        for box in kept_boxes:
            del box["volume"]
            bounding_boxes.append(box)
        
        bbox_path = self.workspace_dir / "bounding_boxes.json"
        with open(bbox_path, 'w') as f:
            json.dump(bounding_boxes, f, indent=4)
            
        print(f"Fusion complete. Saved Mesh and {len(bounding_boxes)} bounding boxes to {self.workspace_dir}")

if __name__ == "__main__":
    pass
