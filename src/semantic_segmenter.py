import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

class SemanticSegmenter:
    def __init__(self, workspace_dir: str = "workspace", model_name: str = "yolov8m-seg.pt"):
        self.workspace_dir = Path(workspace_dir)
        self.images_dir = self.workspace_dir / "images"
        self.masks_dir = self.workspace_dir / "semantics"
        
        # Initialize YOLO segmentation model
        print(f"Loading Semantic Segmentation Model: {model_name}...")
        self.model = YOLO(model_name)
        
    def extract_semantics(self):
        """
        Runs YOLOv8-seg on all images.
        Saves a combined semantic mask as .npy, where each pixel has a class ID.
        Also saves visualization images.
        """
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        
        image_paths = sorted(list(self.images_dir.glob("*.jpg")))
        print(f"Extracting semantics for {len(image_paths)} images...")
        
        for img_path in tqdm(image_paths):
            image = cv2.imread(str(img_path))
            
            # Run inference with higher confidence threshold to avoid false positives
            results = self.model(image, verbose=False, device='cpu', conf=0.7)
            result = results[0]
            
            h, w = image.shape[:2]
            semantic_mask = np.full((h, w), -1, dtype=np.int16) # -1 means background
            
            if result.masks is not None:
                # result.masks.data has shape (N, H, W)
                # result.boxes.cls has shape (N)
                masks = result.masks.data.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                
                # Resize masks if needed
                for i in range(len(masks)):
                    mask = masks[i]
                    cls_id = classes[i]
                    # Resize mask to original image size
                    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    # Assign class ID where mask > 0.5
                    semantic_mask[mask_resized > 0.5] = cls_id
                    
            # Save the semantic mask
            out_path = self.masks_dir / f"{img_path.stem}.npy"
            np.save(str(out_path), semantic_mask)
            
        print(f"Saved semantic masks to {self.masks_dir}")

if __name__ == "__main__":
    # segmenter = SemanticSegmenter()
    # segmenter.extract_semantics()
    pass
