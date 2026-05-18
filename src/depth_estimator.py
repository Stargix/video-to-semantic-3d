import torch
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
from pathlib import Path
from tqdm import tqdm

class DepthEstimator:
    def __init__(self, workspace_dir: str = "workspace", model_name: str = "depth-anything/Depth-Anything-V2-Small-hf"):
        self.workspace_dir = Path(workspace_dir)
        self.images_dir = self.workspace_dir / "images"
        self.depth_dir = self.workspace_dir / "depths"
        
        # Initialize pipeline
        device = -1 # Force CPU due to unsupported RTX 5070 (sm_120) by current PyTorch binaries
        print(f"Loading Depth Estimation Model: {model_name} on device {device}...")
        self.pipe = pipeline(task="depth-estimation", model=model_name, device=device)
        
    def estimate_depths(self):
        """
        Runs Depth estimation on all images in the images_dir.
        Saves the relative depth maps as .npy files.
        """
        self.depth_dir.mkdir(parents=True, exist_ok=True)
        
        image_paths = sorted(list(self.images_dir.glob("*.jpg")))
        print(f"Estimating depth for {len(image_paths)} images...")
        
        for img_path in tqdm(image_paths):
            image = Image.open(img_path).convert("RGB")
            
            # Predict
            result = self.pipe(image)
            # The pipeline returns {"predicted_depth": tensor/array, "depth": PIL Image}
            depth_tensor = result["predicted_depth"]
            
            # Convert to numpy and resize to original image size if needed
            # HF pipeline usually handles resizing back, but let's be sure
            if isinstance(depth_tensor, torch.Tensor):
                depth_map = depth_tensor.squeeze().cpu().numpy()
            else:
                depth_map = np.array(depth_tensor)
                
            # Resize depth map to match input image exactly
            # (sometimes the pipeline output is slightly different due to padding/model architecture)
            depth_map = cv2.resize(depth_map, (image.width, image.height), interpolation=cv2.INTER_LINEAR)
            
            # Save as .npy
            out_path = self.depth_dir / f"{img_path.stem}.npy"
            np.save(str(out_path), depth_map)
            
        print(f"Saved depth maps to {self.depth_dir}")

if __name__ == "__main__":
    # estimator = DepthEstimator()
    # estimator.estimate_depths()
    pass
