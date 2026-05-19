import os
from pathlib import Path
import huggingface_hub

# Hugging Face dataset ID and the 10 selected ScanNet room sequences
REPO_ID = "Liuff23/scannet-videos"
REPO_TYPE = "dataset"

SCENES = [
    "scene0019_00.mp4",
    "scene0077_00.mp4",
    "scene0081_00.mp4",
    "scene0083_00.mp4",
    "scene0083_01.mp4",
    "scene0088_03.mp4",
    "scene0090_00.mp4",
    "scene0104_00.mp4",
    "scene0112_02.mp4",
    "scene0117_00.mp4",
]

def main():
    script_dir = Path(__file__).resolve().parent
    experiments_dir = script_dir.parent
    videos_dir = experiments_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    
    print("==================================================")
    print("      ScanNet Indoor Room Videos Downloader       ")
    print("==================================================")
    print(f"Target repository: {REPO_ID}")
    print(f"Destination:       {videos_dir}\n")
    
    successful_downloads = 0
    for scene in SCENES:
        dest_path = videos_dir / scene
        if dest_path.exists():
            print(f"[✓] {scene} already exists. Skipping.")
            successful_downloads += 1
            continue
            
        print(f"Downloading {scene}...")
        try:
            huggingface_hub.hf_hub_download(
                repo_id=REPO_ID,
                filename=scene,
                repo_type=REPO_TYPE,
                local_dir=str(videos_dir),
                local_dir_use_symlinks=False
            )
            print(f"[✓] Successfully downloaded {scene}")
            successful_downloads += 1
        except Exception as e:
            print(f"[!] Error downloading {scene}: {e}")
            
    print("==================================================")
    print(f"Successfully verified/downloaded {successful_downloads}/{len(SCENES)} videos.")
    print("==================================================")

if __name__ == "__main__":
    main()
