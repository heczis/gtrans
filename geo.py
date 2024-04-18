"""
Simple geometry tools
"""
import numpy as np

def is_in_half_plane(origin, direction, point, zero_tol=1e-8):
    return direction.dot(point - origin) >= 0

def is_in_half_plane_by_pts_2d(pt1, pt2, point, zero_tol=1e-8):
    origin = pt1
    direction = np.array([[0, 1], [-1, 0]]).dot(pt2 - pt1)
    return is_in_half_plane(origin, direction, point, zero_tol)
