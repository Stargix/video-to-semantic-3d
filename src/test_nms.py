import numpy as np

def test_inside():
    c1 = np.array([0.5, 0, 0])
    c2 = np.array([0, 0, 0])
    R2 = np.eye(3)
    ext2 = np.array([2.0, 2.0, 2.0])
    
    local_c1 = R2.T @ (c1 - c2)
    is_inside = np.all(np.abs(local_c1) <= (ext2 / 2.0) * 1.2)
    print("Inside:", is_inside)

test_inside()
