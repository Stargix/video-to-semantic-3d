# Video to Semantic 3D Reconstruction

This repository contains the complete implementation for the **Intern Challenge: From Video to 3D Reconstruction**. The system processes a short casual video of an indoor area (e.g., captured on a phone) and reconstructs a geometrically coherent, semantically labeled 3D scene in a single execution pipeline.

---

## Challenge Objectives and Core Goals

### Core Goal: Geometric Reconstruction
* **Objective**: Generate a geometrically coherent and consistent 3D representation of an indoor scene from a casual handheld video.
* **Implementation**: We recover high-fidelity geometric structures and correct physical scale by anchoring dense deep depth maps to sparse metric keypoints, fused into a watertight 3D mesh via Truncated Signed Distance Function (TSDF) Volume Integration.

### Optional Extensions: Semantic Alignment
* **Objective**: Assign 3D semantic labels (e.g., chairs, couch, refrigerator) aligned with the underlying scene geometry.
* **Implementation**: The system performs 2D instance segmentation on each frame, back-projects the labels into 3D metric coordinates, and filters spatial noise using Density-Based Spatial Clustering (DBSCAN) to fit tight, accurate 3D Oriented Bounding Boxes (OBB).

### Performance Approach
* There are no constraints on real-time performance. The system prioritizes geometric precision, structural consistency, and robust alignment over speed.

---

## Technical Approach and System Design

The reconstruction pipeline is written entirely in Python and operates through five sequential stages:

```
[Casual Video Input]
         │
         ▼
 1. Frame Extraction (VideoProcessor)
         │
         ▼
 2. Sparse SfM Tracking (SfMMapper) ──► Recovers camera poses & metric keypoints
         │
         ▼
 3. Dense Depth Prior (DepthEstimator) ──► Evaluates relative depth (DepthAnythingV2)
         │
         ▼
 4. Instance Semantics (SemanticSegmenter) ──► Computes 2D segmentations (YOLOv8-seg)
         │
         ▼
 5. Scale Alignment & TSDF Integration (FusionEngine) ──► Generates 3D color mesh
         │
         ▼
 6. DBSCAN Spatial Clustering (FusionEngine) ──► Fits 3D Oriented Bounding Boxes (OBB)
```

### 1. Sparse Geometric Consistency
Traditional dense multi-view stereo often fails on low-texture walls and reflective surfaces typical of indoor rooms. To establish a reliable geometric foundation, we use `pycolmap` to perform sparse Structure-from-Motion (SfM). This step recovers:
* Highly accurate camera intrinsic and extrinsic trajectories.
* A metric, scale-consistent sparse keypoint cloud of the room.

### 2. Dense Depth Priors and Metric scale Alignment
To obtain dense coverage, we estimate relative depth for each extracted frame using `DepthAnythingV2`. Because feed-forward depth networks output depth up to an arbitrary scale and shift, we formulate a scale-and-shift alignment problem. For each frame, we match unprojected depth points to the corresponding sparse COLMAP keypoints and solve a robust least-squares optimization:
$$\min_{s, t} \sum_{i} \left\| (s \cdot d_{i,\text{rel}} + t) - d_{i,\text{metric}} \right\|_2$$
This anchors the deep dense depth maps to the physical metric scale of the room.

### 3. Color TSDF Volume Integration
Instead of exporting loose, noisy point clouds, we integrate the scale-aligned dense depth maps and RGB frames into a unified Truncated Signed Distance Function (TSDF) volume using Open3D. This volumetric integration:
* Automatically resolves surface redundancies and noise.
* Smooths out high-frequency sensor errors.
* Reconstructs a watertight, consistent 3D color mesh (`scene_mesh.ply`).

### 4. Dense 3D Semantic Projection and Clustering
The system runs `YOLOv8-seg` on the input frames to extract 2D instance masks. These masks are back-projected into 3D space using the metric depth maps. To transition from noisy, point-wise classifications to structured 3D object representations, we:
* Filter the semantic point cloud using statistical outlier removal.
* Cluster the points for each semantic category independently using DBSCAN.
* Extract the physical boundaries of each detected object and fit a 3D Oriented Bounding Box (OBB) described by its 3D center, extents, and a 3x3 rotation matrix.
* Save the final detections to a lightweight JSON metadata file (`bounding_boxes.json`).

---

## Dining Room (Comedor) Demonstration Results

The default demonstration uses a casual video of a dining room (`inputs/demo3.mp4`). The pipeline automatically maps the geometric layout of the room, reconstructing structural surfaces and extracting oriented bounding boxes for the dining chairs and surrounding furniture. 

The output directory (`outputs/demo_output_<timestamp>/`) includes:
* `scene_mesh.ply`: The reconstructed 3D watertight color mesh.
* `bounding_boxes.json`: Oriented bounding box dimensions, classes, and orientations for detected furniture.
* `pipeline_summary.png`: A comprehensive diagnostic dashboard illustrating the recovered camera trajectory and execution times.
* `view_general.png`: A high-fidelity, automatically captured 3D screenshot showcasing the reconstructed room geometry overlaid with the 3D Oriented Bounding Boxes (OBB).

Below is the automatically captured 3D visualization showing the dining room reconstruction overlaid with semantically labeled bounding boxes:

<p align="center">
  <img src="docs/demo_room_view.png" alt="3D Reconstruction Visual Output for Dining Room (Comedor) Demo" width="800">
</p>

Below is the corresponding pipeline diagnostic summary dashboard:

<p align="center">
  <img src="docs/pipeline_summary.png" alt="3D Reconstruction Pipeline Summary for Dining Room (Comedor) Demo" width="800">
</p>

---

## Quantitative Reconstruction Experiments

To evaluate the pipeline's generalization and metric accuracy under controlled settings, we ran reconstructions across 5 different indoor room videos from the ScanNet dataset at a uniform frame rate of 2.0 FPS.

The pipeline successfully matched features, calculated camera trajectories, estimated dense depth, aligned them metrically, and exported highly accurate 3D semantic bounding boxes.

| Room Video (ScanNet) | File Size | Frame Count (Extracted) | 3D Objects Mapped in Bounding Boxes | Preserved Visuals |
| :--- | :--- | :--- | :--- | :--- |
| **`scene0104_00`** | 2.99 MB | 87 | `chair`, `toaster`, `refrigerator` | `pipeline_summary.png`, [view_general.png](docs/scene0104_00_view.png) |
| **`scene0019_00`** | 9.84 MB | 49 | `refrigerator`, `couch` | `pipeline_summary.png`, [view_general.png](docs/scene0019_00_view.png) |
| **`scene0090_00`** | 9.79 MB | 38 | `sink`, `toilet` | `pipeline_summary.png`, [view_general.png](docs/scene0090_00_view.png) |
| **`scene0112_02`** | 10.24 MB | 42 | `sink`, `toilet` | `pipeline_summary.png`, [view_general.png](docs/scene0112_02_view.png) |
| **`scene0117_00`** | 9.30 MB | 52 | `refrigerator` | `pipeline_summary.png`, [view_general.png](docs/scene0117_00_view.png) |

### Visual Gallery of Reconstructed ScanNet Scenes

The following visual outputs display the reconstructed 3D watertight meshes integrated with the 3D Oriented Bounding Boxes (OBB) fitted automatically via DBSCAN clustering for the 5 successful sequences:

#### 1. Scene0104_00 (Kitchen & Dining Area)
<p align="center">
  <img src="docs/scene0104_00_view.png" alt="3D Reconstruction and OBB fitting for scene0104_00" width="800">
</p>

#### 2. Scene0019_00 (Living Room with Couch)
<p align="center">
  <img src="docs/scene0019_00_view.png" alt="3D Reconstruction and OBB fitting for scene0019_00" width="800">
</p>

#### 3. Scene0090_00 (Bathroom Layout)
<p align="center">
  <img src="docs/scene0090_00_view.png" alt="3D Reconstruction and OBB fitting for scene0090_00" width="800">
</p>

#### 4. Scene0112_02 (Bathroom Layout - Alternative Perspective)
<p align="center">
  <img src="docs/scene0112_02_view.png" alt="3D Reconstruction and OBB fitting for scene0112_02" width="800">
</p>

#### 5. Scene0117_00 (Kitchen Layout)
<p align="center">
  <img src="docs/scene0117_00_view.png" alt="3D Reconstruction and OBB fitting for scene0117_00" width="800">
</p>


---

## Recording Guidelines for Optimal Reconstruction

For the best possible 3D reconstruction when recording custom video sequences on a mobile device:
1. **Slow and Steady Camera Motion**: Rapid panning causes motion blur, which disrupts keypoint tracking and feature matching in COLMAP.
2. **Smooth, Non-Rotational Trajectory**: Walk or move the camera in a smooth linear or orbital arc. Avoid standing in a single spot and rotating the camera (pure rotation lacks parallax, preventing depth triangulation).
3. **Balanced Lighting**: Avoid aiming the camera directly at bright windows or intense light sources. Well-distributed lighting ensures stable visual features.
4. **High Visual Overlap**: Move slowly enough to maintain high overlap (80%+) between consecutive frames.
5. **Duration**: Keep videos short and concise (5 to 15 seconds) to limit tracking drift and processing times.

---

## Installation and Environment Setup

### Prerequisites
* Operating System: Linux (or Windows via WSL2)
* Python 3.10 or higher
* NVIDIA GPU (recommended for accelerated depth and semantic estimation)

### Dependencies
Create and activate the environment using Conda:
```bash
conda env create -f environment.yml
conda activate vid3d
```
Alternatively, install dependencies via `pip`:
```bash
pip install -r requirements.txt
```

---

## Execution Pipeline

### 1. Run the Full Reconstruction
To execute the entire pipeline (frame extraction, SfM, depth estimation, semantic segmentation, metric fusion, and OBB extraction) in one command:
```bash
python src/main.py path/to/your/video.mp4 --workspace my_reconstruction --fps 2.0
```

### 2. Visualize Interactive 3D Results
To launch the interactive Open3D visualizer showing the reconstructed scene and fitted semantic bounding boxes:
```bash
python src/visualize.py --workspace my_reconstruction
```
* **Interactive Visualizer Controls**:
  * Left Click + Drag: Rotate the camera
  * Right Click + Drag: Translate the camera
  * Scroll Wheel: Zoom in / out
  * Press `[Q]`: Close the visualizer

---

## Project Component Directory

* `src/main.py`: Main CLI entrypoint to orchestrate the pipeline stages.
* `src/video_processor.py`: Extracts and filters video frames.
* `src/sfm_mapper.py`: Orchestrates pycolmap for camera pose estimation.
* `src/depth_estimator.py`: Executes DepthAnythingV2 for relative depth inference.
* `src/semantic_segmenter.py`: Performs YOLOv8-seg instance segmentation.
* `src/fusion_engine.py`: Handles metric scale alignment, TSDF integration, DBSCAN clustering, and bounding box fitting.
* `src/visualize.py`: Headless and interactive 3D visualizer.
