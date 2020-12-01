"""Math conversion utility functions."""

import numpy as np
from transforms3d.axangles import axangle2mat, mat2axangle
from transforms3d.euler import euler2mat, mat2euler
from transforms3d.quaternions import mat2quat, quat2mat

__all__ = ('mat2rvec', 'rvec2mat', 'joint2mat', 'mat2joint', 'mat2pb', 'pb2mat')


def mat2rvec(mat):
    """Convert rotation matrix to rotation vector."""
    axis, angle = mat2axangle(mat, unit_thresh=1e-05)
    return axis * angle


def rvec2mat(rvec):
    """Convert rotation vector to rotation matrix."""
    angle = np.linalg.norm(rvec)
    axis = rvec if angle != 0.0 else [0.0, 0.0, 1.0]
    mat = axangle2mat(axis, angle)
    return mat


def joint2mat(axes, angles):
    """Compose rotation matrix of a multi-dof joint.

    Arguments:
        axes {str} -- joint axes
        angles {list} -- joint angles

    Returns:
        array -- rotation matrix
    """
    rest = ''.join([i for i in 'xyz' if i not in axes])
    euler = [0.0, 0.0, 0.0]
    euler[:len(axes)] = angles[:len(axes)]
    return euler2mat(*euler, 'r' + axes + rest)


def mat2joint(mat, axes):
    """Decompose rotation matrix of a multi-dof joint.

    Arguments:
        mat {mat3} -- rotation matrix
        axes {str} -- joint axes

    Returns:
        list -- rotation angles about joint axes
    """
    rest = ''.join([i for i in 'xyz' if i not in axes])
    euler = mat2euler(mat, 'r' + axes + rest)
    return euler[:len(axes)]


def mat2pb(mat):
    """Convert rotation matrix to quaternion in PyBullet's format."""
    quat = mat2quat(mat)
    return quat[1], quat[2], quat[3], quat[0]


def pb2mat(orn):
    """Convert quaternion in PyBullet's format to rotation matrix."""
    quat = orn[3], orn[0], orn[1], orn[2]
    return quat2mat(quat)
