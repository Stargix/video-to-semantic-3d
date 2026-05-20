import shutil
from pathlib import Path
import sys

import subprocess

def clean_workspace(room_name: str, workspace_dir: Path, results_dir: Path):
    """
    Copies key lightweight outputs (graphs, gifs, bounding boxes, meshes) to results_dir,
    and then deletes the heavy temporary workspace to save space.
    """
    print(f"\n--- Cleaning up workspace for {room_name} ---")
    
    # 1. Generate the high-quality 3D render photo.
    # We first try to use Windows python (py.exe) with src/visualize.py --screenshot-only because:
    #   - WSL offscreen rendering segfaults or produces low-quality, squashed dark images.
    #   - Windows has native GPU acceleration and renders the beautiful white-background interactive view instantly.
    rendered_successfully = False
    
    scripts_dir = Path(__file__).resolve().parent
    repo_root = scripts_dir.parent.parent
    
    try:
        relative_workspace = workspace_dir.relative_to(repo_root)
    except ValueError:
        relative_workspace = workspace_dir

    print("Attempting high-quality rendering using Windows py.exe visualizer...")
    try:
        # Run py.exe with screenshot-only flag, using relative paths and repo_root as CWD
        res = subprocess.run(
            ["py.exe", "src/visualize.py", "--workspace", str(relative_workspace), "--screenshot-only"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=60
        )
        if res.returncode == 0:
            print("[✓] Windows py.exe rendering completed successfully.")
            rendered_successfully = True
        else:
            print(f"[!] Windows py.exe rendering failed with exit code {res.returncode}.")
            if res.stderr:
                print(f"Windows py.exe stderr:\n{res.stderr.strip()}")
    except Exception as e:
        print(f"[!] Could not run Windows py.exe rendering: {e}")

    # Fallback to headless offscreen renderer in WSL if Windows py.exe fails or is unavailable
    if not rendered_successfully:
        render_script = scripts_dir / "generate_render_photo.py"
        if render_script.exists():
            try:
                print("Falling back to WSL headless offscreen render generation...")
                res = subprocess.run(
                    [sys.executable, str(render_script), "--workspace", str(workspace_dir)],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if res.returncode == 0:
                    print("[✓] WSL headless render completed successfully.")
                    rendered_successfully = True
                else:
                    print(f"[!] WSL headless render failed (exit code {res.returncode}). Continuing cleanup anyway.")
                    if res.stderr:
                        print(f"Subprocess stderr:\n{res.stderr.strip()}")
            except Exception as e:
                print(f"[!] Failed to run WSL headless render process: {e}")
            
    # 2. Create target results directory
    room_results_dir = results_dir / room_name
    room_results_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Define the files we want to preserve
    files_to_preserve = [
        "pipeline_summary.png",
        "3d_reconstruction.gif",
        "bounding_boxes.json",
        "scene_mesh.ply",
        "scene_rgb.ply",
        "view_general.png"
    ]
    
    preserved_count = 0
    for filename in files_to_preserve:
        src_file = workspace_dir / filename
        if src_file.exists():
            dst_file = room_results_dir / filename
            shutil.copy2(src_file, dst_file)
            print(f"[✓] Preserved: {filename} -> {dst_file.relative_to(results_dir.parent.parent)}")
            preserved_count += 1
        else:
            print(f"[!] Warning: Expected output file {filename} not found in workspace.")
            
    # 3. Completely delete the temporary workspace folder
    if workspace_dir.exists():
        print(f"Purging heavy temporary workspace folder: {workspace_dir}...")
        try:
            shutil.rmtree(workspace_dir)
            print(f"[✓] Workspace purged successfully.")
        except Exception as e:
            print(f"[!] Error purging workspace: {e}")
            
    print(f"--- Finished cleaning for {room_name} (Preserved {preserved_count} files) ---\n")
    return preserved_count > 0

def main():
    # If run directly as a script, clean up all workspaces
    script_dir = Path(__file__).resolve().parent
    temp_dir = script_dir.parent / "temp_workspaces"
    results_dir = script_dir.parent / "results"
    
    if not temp_dir.exists() or not any(temp_dir.iterdir()):
        print("No temporary workspaces found to clean.")
        return
        
    print("==================================================")
    print("      Automated Results Preservation & Cleaner    ")
    print("==================================================")
    
    for room_dir in temp_dir.iterdir():
        if room_dir.is_dir():
            clean_workspace(room_dir.name, room_dir, results_dir)
            
    print("Manual cleanup completed.")

if __name__ == "__main__":
    main()
