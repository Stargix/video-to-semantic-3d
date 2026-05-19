# 3D Semantic Reconstruction Room Experimentation Suite

This directory contains a complete, automated experimentation harness for running your **Video to Semantic 3D Reconstruction** pipeline on casual indoor room video sequences. 

To ensure the casual handheld camera processing capabilities of your pipeline are rigorously evaluated on authentic indoor residential layouts (rather than dynamic transit spaces like shopping malls), we pivoted to the **ScanNet Videos dataset** (`Liuff23/scannet-videos` on Hugging Face).

---

## 📂 Project Structure

```
experiments/
├── README.md                    # This documentation of experimental setup and results
├── videos/                      # [GIT-IGNORED] Storage for the 10 downloaded .mp4 videos
├── results/                     # Git-tracked lightweight results (graphs, bounding boxes)
│   ├── scene0019_00/
│   │   ├── pipeline_summary.png # Camera trajectory plot, sparse points, and execution times
│   │   └── bounding_boxes.json  # JSON storing center, rotation matrix R, and extent of 3D objects
│   └── ... (results for all 10 scenes)
└── scripts/                     # Python scripts managing the experiment lifecycle
    ├── download_dataset.py      # Programmatic downloader using huggingface_hub
    ├── run_experiments.py       # Sequential runner with intelligent caching/resume features
    └── clean_outputs.py         # Results harvester that copied graphs and purged heavy temp directories
```

---

## 📊 Summary of 3D Mapping Results

We ran the reconstruction pipeline across **10 different indoor rooms** using a uniform framerate of **`2.0 FPS`** for video frame extraction.

The pipeline successfully matched features, calculated camera trajectories via COLMAP Structure from Motion (SfM), estimated dense depth via DepthAnythingV2, aligned them metrically, integrated them into a TSDF volume, and exported **11 high-confidence 3D semantic bounding boxes**.

| Room Video (ScanNet) | File Size | Frame Count (Extracted) | 3D Objects Mapped in Bounding Boxes | Preserved Visuals |
| :--- | :--- | :--- | :--- | :--- |
| **`scene0104_00`** | 2.99 MB | 87 | 🪑 `chair`, 🍞 `toaster`, ❄️ `refrigerator` | `pipeline_summary.png` |
| **`scene0019_00`** | 9.84 MB | 49 | ❄️ `refrigerator`, 🛋️ `couch` | `pipeline_summary.png` |
| **`scene0077_00`** | 5.78 MB | 13 | *(None detected)* | `pipeline_summary.png` |
| **`scene0081_00`** | 6.45 MB | 26 | *(None detected)* | `pipeline_summary.png` |
| **`scene0083_00`** | 9.90 MB | 16 | *(None detected)* | `pipeline_summary.png` |
| **`scene0083_01`** | 9.89 MB | 29 | *(None detected)* | `pipeline_summary.png` |
| **`scene0088_03`** | 5.70 MB | 36 | 🪑 `chair` | `pipeline_summary.png` |
| **`scene0090_00`** | 9.79 MB | 38 | 🚰 `sink`, 🚽 `toilet` | `pipeline_summary.png` |
| **`scene0112_02`** | 10.24 MB | 42 | 🚰 `sink`, 🚽 `toilet` | `pipeline_summary.png` |
| **`scene0117_00`** | 9.30 MB | 52 | ❄️ `refrigerator` | `pipeline_summary.png` |

---

## 🚀 How to Run the Experiments

### 1. Pre-requisites & Environment
Activate the Conda environment:
```bash
conda activate vid3d
```

### 2. Download the Room Videos
To fetch the 10 ScanNet room videos (totaling **~79 MB**):
```bash
python experiments/scripts/download_dataset.py
```
This script downloads the optimized video files directly from Hugging Face Hub using programmatic APIs.

### 3. Run Reconstructions
To process all downloaded videos at a target frame rate (e.g. 2.0 FPS) sequentially:
```bash
python experiments/scripts/run_experiments.py --fps 2.0
```
* **Intelligent Caching**: The orchestrator automatically checks your `results/` folder. If a video's `pipeline_summary.png` and `bounding_boxes.json` are already successfully generated, it will immediately skip it and resume processing the next queued scene.
* **Auto-Cleanup**: After finishing each reconstruction, the system automatically harvests the lightweight plots and metadata to `results/` and purges the heavy `temp_workspaces/` directories (saving multiple gigabytes of disk space).

---

## 🖼️ Structure of Lightweight Outputs

For each reconstructed room under `results/<room_id>/`, the following files are preserved:
1. **`pipeline_summary.png`**: Contains a 3-part layout:
   * Left: 2D representation of the estimated camera flight path and reconstructed 3D keypoint cloud.
   * Right-Top: Step-by-step pipeline execution times.
   * Right-Bottom: Sparse point distribution and statistical summary.
2. **`bounding_boxes.json`**: Describes 3D oriented bounding boxes for all detected objects:
   ```json
   [
     {
       "class_id": 56,
       "class_name": "chair",
       "center": [-0.83, -3.08, -5.49],
       "R": [
         [0.76, -0.27, 0.58],
         [0.18, 0.96, 0.20],
         [-0.61, -0.04, 0.78]
       ],
       "extent": [1.80, 2.90, 1.42]
     }
   ]
   ```
