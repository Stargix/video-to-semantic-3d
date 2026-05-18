import cv2
import os
import shutil
from pathlib import Path

class VideoProcessor:
    def __init__(self, output_dir: str = "workspace"):
        self.output_dir = Path(output_dir)
        self.frames_dir = self.output_dir / "images"
        
    def extract_frames(self, video_path: str, fps: float = 3.0, resize_max_dim: int = 1024) -> str:
        """
        Extracts frames from a video at a specified FPS.
        
        Args:
            video_path: Path to the input video.
            fps: Desired frames per second to extract.
            resize_max_dim: Maximum dimension (width or height) to resize the frames to.
                            Keeps aspect ratio. Helps speed up processing and saves memory.
                            
        Returns:
            Path to the directory containing the extracted frames.
        """
        print(f"Extracting frames from {video_path} at {fps} FPS...")
        
        # Clean up existing frames
        if self.frames_dir.exists():
            shutil.rmtree(self.frames_dir)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file {video_path}")
            
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(round(video_fps / fps))
        if frame_interval < 1:
            frame_interval = 1
            
        frame_idx = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_interval == 0:
                # Resize if necessary
                h, w = frame.shape[:2]
                if max(h, w) > resize_max_dim:
                    scale = resize_max_dim / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    
                output_path = self.frames_dir / f"frame_{saved_count:05d}.jpg"
                cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved_count += 1
                
            frame_idx += 1
            
        cap.release()
        print(f"Extracted {saved_count} frames to {self.frames_dir}")
        return str(self.frames_dir)

if __name__ == "__main__":
    # Test
    processor = VideoProcessor()
    # processor.extract_frames("test.mp4")
