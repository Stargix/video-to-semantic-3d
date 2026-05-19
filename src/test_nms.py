import numpy as np
import cv2
import open3d as o3d

def get_y_vertical_obb(pts):
    pts_xz = pts[:, [0, 2]].astype(np.float32)
    rect = cv2.minAreaRect(pts_xz)
    box2d = cv2.boxPoints(rect)
    
    # rect[0] is center, rect[1] is width, height, rect[2] is angle in degrees
    # We need the 3D center, R, extent
    # Extent in X and Z is from rect[1]
    ext_x, ext_z = rect[1]
    ext_y = pts[:, 1].max() - pts[:, 1].min()
    
    center_y = (pts[:, 1].max() + pts[:, 1].min()) / 2.0
    center_x, center_z = rect[0]
    
    # Angle in degrees
    angle_rad = np.deg2rad(rect[2])
    # The rotation matrix for minAreaRect:
    # cv2.minAreaRect angle is the angle between the X axis and the first side of the rectangle.
    # It rotates the rectangle around its center.
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    R = np.array([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a]
    ])
    
    # Because cv2 angles might flip width/height, we just use the box directly:
    return [center_x, center_y, center_z], R.tolist(), [ext_x, ext_y, ext_z]

pts = np.random.rand(100, 3)
center, R, extent = get_y_vertical_obb(pts)
print("Center:", center)
print("R:\n", np.array(R))
print("Extent:", extent)
