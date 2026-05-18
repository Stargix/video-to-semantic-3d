import os
os.environ["GDK_BACKEND"] = "x11"
os.environ["XDG_SESSION_TYPE"] = "x11"
import open3d as o3d
import open3d.visualization.gui as gui

print(hasattr(o3d.visualization, 'O3DVisualizer'))
