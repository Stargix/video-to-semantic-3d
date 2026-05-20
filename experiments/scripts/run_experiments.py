import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add script directory to Python path to import clean_workspace
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir))
from clean_outputs import clean_workspace

def run_single_experiment(video_path: Path, temp_workspace: Path, results_dir: Path, fps: float):
    room_name = video_path.stem
    print("=" * 60)
    print(f"Starting 3D Semantic Reconstruction for: {room_name}")
    print(f"Input Video: {video_path}")
    print(f"Workspace:   {temp_workspace}")
    print(f"Frame Rate:  {fps} FPS")
    print("=" * 60)
    
    # Run the main pipeline
    cmd = [
        sys.executable, "src/main.py",
        str(video_path),
        "--workspace", str(temp_workspace),
        "--fps", str(fps)
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run and stream output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream the stdout of the pipeline to the terminal in real-time
        if process.stdout:
            for line in process.stdout:
                print(f"[{room_name}] {line}", end="")
                
        process.wait()
        
        if process.returncode == 0:
            print(f"\n[✓] Reconstruction pipeline completed successfully for {room_name}!")
            # Clean up immediately to save disk space
            clean_workspace(room_name, temp_workspace, results_dir)
            return True
        else:
            print(f"\n[!] Error: Reconstruction pipeline failed for {room_name} with exit code {process.returncode}")
            return False
            
    except Exception as e:
        print(f"\n[!] Error running pipeline for {room_name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="3D Semantic Reconstruction Experiment Runner")
    parser.add_argument("--fps", type=float, default=2.0, help="Frames per second to extract (default: 2.0)")
    parser.add_argument("--video-id", type=str, default=None, help="Process a single specific video (e.g. advio_01)")
    args = parser.parse_args()
    
    videos_dir = script_dir.parent / "videos"
    temp_dir = script_dir.parent / "temp_workspaces"
    results_dir = script_dir.parent / "results"
    
    temp_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Glob for both .mp4 and .mov files
    video_paths = sorted(list(videos_dir.glob("*.mp4")) + list(videos_dir.glob("*.mov")))
    
    if not video_paths:
        print(f"No videos found in {videos_dir}. Please run download_dataset.py first.")
        return
        
    # If a specific video is requested, filter the list
    if args.video_id:
        video_paths = [v for v in video_paths if args.video_id.lower() in v.stem.lower()]
        if not video_paths:
            print(f"Requested video '{args.video_id}' not found in {videos_dir}.")
            return
            
    print("==================================================")
    print("       3D Semantic Reconstruction Experiment Runner ")
    print("==================================================")
    print(f"Found {len(video_paths)} videos to reconstruct.")
    
    success_count = 0
    for video_path in video_paths:
        room_name = video_path.stem
        room_results_dir = results_dir / room_name
        if (room_results_dir / "pipeline_summary.png").exists() and ((room_results_dir / "scene_mesh.ply").exists() or (room_results_dir / "scene_rgb.ply").exists()):
            print(f"[✓] {room_name} is already successfully reconstructed. Skipping.")
            success_count += 1
            continue
            
        temp_workspace = temp_dir / room_name
        if run_single_experiment(video_path, temp_workspace, results_dir, args.fps):
            success_count += 1
            
    print("==================================================")
    print("           All Experiments Completed              ")
    print("==================================================")
    print(f"Successfully processed {success_count}/{len(video_paths)} videos.")

if __name__ == "__main__":
    main()
