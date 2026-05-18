# Video to Semantic 3D Reconstruction

This repository provides a complete, 100% Python pipeline to convert a casual indoor video into a dense, geometrically coherent 3D point cloud with aligned semantic labels.

## Approach & Design Choices (SOTA Validation)

Our approach follows the cutting-edge trend seen in modern 3D vision research (such as ByteDance's **Depth-Anything-3 COLMAP integration** and other "feed-forward" 3D models). Rather than relying on slow Multi-View Stereo (MVS) or compiling fragile CUDA kernels for 3D Gaussian Splatting, we leverage Foundation Models alongside robust Structure-from-Motion (SfM):
1. **Sparse Geometric Consistency**: We use `pycolmap` (SfM) to accurately recover camera poses and a metrically consistent sparse point cloud.
2. **Dense Semantic & Depth Priors**: We use `DepthAnythingV2` for incredibly dense and high-quality relative depth estimation (solving COLMAP's failure on textureless indoor walls), and `YOLOv8-seg` for robust 2D instance segmentation.
3. **Metric Alignment & Fusion**: We dynamically align the dense relative depth maps to the sparse COLMAP metric points. We then unproject these dense maps into a 3D point cloud.
4. **3D Object Detection (DBSCAN)**: Instead of generating messy point-wise semantic colors, we apply spatial density clustering (DBSCAN) to the semantic points to extract pure, robust **Oriented 3D Bounding Boxes (OBB)**.

**Why this is State-of-the-Art for Usability:**
- **Zero custom CUDA compilation**: Unlike 3DGS or NeRFs, it runs purely in Python and avoids PyTorch/CUDA version mismatches.
- **Geometrically Coherent**: The scale and shift of the deep depth maps are anchored to the physical metric scale provided by COLMAP.
- **Actionable Semantics**: Outputs clean 3D Object Bounding Boxes (JSON) rather than noisy, un-filtered point labels.

## How to Record Optimal Videos

To ensure the best possible 3D reconstruction, follow these guidelines when recording your input video (e.g., on a phone):
1. **Slow and Steady**: Move the camera slowly. Rapid movements cause motion blur, which destroys the feature matching in COLMAP.
2. **Smooth Trajectory**: Walk in a smooth arc or straight line. Do not stand in one place and simply rotate the camera (pure rotation provides no parallax, making depth triangulation impossible).
3. **Good Lighting**: Ensure the room is well-lit. Avoid aiming the camera directly at bright windows or lights.
4. **Overlap**: Make sure there is high visual overlap between consecutive frames.
5. **Length**: Keep it short (5 to 15 seconds is ideal for an indoor area).

## Requirements

The project uses standard Python packages. You should have Python 3.10+ installed. A GPU is recommended for the deep learning models, but it will fallback to CPU if necessary.

```bash
pip install -r requirements.txt
```

## How to Run

1. **Run the full pipeline** on your video:
   ```bash
   python src/main.py path/to/your/video.mp4 --workspace my_reconstruction --fps 2.0
   ```
   This will automatically extract frames, run SfM, predict depth and semantics, align, and fuse them into a 3D scene.

2. **Visualize the result**:
   ```bash
   python src/visualize.py --workspace my_reconstruction
   ```
   A window will open showing the RGB reconstructed point cloud. Close the window to then see the Semantically colored point cloud!

## Project Structure
- `src/video_processor.py`: Extracts frames from video.
- `src/sfm_mapper.py`: Wrapper around pycolmap for camera tracking.
- `src/depth_estimator.py`: DepthAnythingV2 inference.
- `src/semantic_segmenter.py`: YOLOv8-seg inference.
- `src/fusion_engine.py`: Core logic for scale alignment and 3D unprojection.
- `src/main.py`: CLI to orchestrate the pipeline.
- `src/visualize.py`: Open3D visualization script.
