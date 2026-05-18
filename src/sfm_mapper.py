import os
import pycolmap
from pathlib import Path

class SfMMapper:
    def __init__(self, workspace_dir: str = "workspace"):
        self.workspace_dir = Path(workspace_dir)
        self.images_dir = self.workspace_dir / "images"
        self.db_path = self.workspace_dir / "database.db"
        self.output_path = self.workspace_dir / "sparse"
        
    def run_reconstruction(self) -> pycolmap.Reconstruction:
        """
        Runs COLMAP feature extraction, matching, and incremental mapping.
        Returns the best reconstruction found.
        """
        print("Starting Structure from Motion with pycolmap...")
        
        # Clean up previous runs
        if self.db_path.exists():
            self.db_path.unlink()
        if self.output_path.exists():
            import shutil
            shutil.rmtree(self.output_path)
            
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Feature extraction
        print("Extracting features...")
        pycolmap.extract_features(
            database_path=self.db_path,
            image_path=self.images_dir
        )
        
        # Feature matching
        print("Matching features...")
        pycolmap.match_exhaustive(
            database_path=self.db_path
        )
        
        # Incremental mapping
        print("Running incremental mapping...")
        maps = pycolmap.incremental_mapping(
            database_path=self.db_path,
            image_path=self.images_dir,
            output_path=self.output_path
        )
        
        if not maps:
            raise RuntimeError("COLMAP failed to reconstruct any maps.")
            
        best_map_idx = max(maps, key=lambda i: maps[i].num_reg_images())
        best_map = maps[best_map_idx]
        
        # Save the best map to a known predictable location for the fusion engine
        import shutil
        shutil.rmtree(self.output_path) # Remove the 0, 1 subfolders
        self.output_path.mkdir(parents=True, exist_ok=True)
        best_map.write(self.output_path)
        
        print(f"Reconstruction complete. Reconstructed {best_map.num_reg_images()} images with {best_map.num_points3D()} 3D points.")
        
        return best_map

if __name__ == "__main__":
    # Test
    # mapper = SfMMapper()
    # mapper.run_reconstruction()
    pass
