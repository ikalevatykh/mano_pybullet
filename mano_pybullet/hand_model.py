"""Hand models."""

import collections

import numpy as np

from .mano_model import ManoModel
from .math_utils import joint2mat, mat2joint, mat2rvec, rvec2mat

__all__ = ('HandModel', 'HandModel20', 'HandModel45')

Joint = collections.namedtuple('Joint', ['origin', 'basis', 'axes', 'limits'])


class HandModel(ManoModel):
    """Base rigid hand model.

    The model provides rigid hand kinematics description and math_utils
    from the joint space to the MANO model pose.
    """

    def __init__(self, left_hand=False):
        """Initialize a HandModel.

        Keyword Arguments:
            left_hand {bool} -- create a left hand model (default: {False})
        """
        super().__init__(left_hand)
        self._joints = self._make_joints()
        self._basis = [joint.basis for joint in self._joints]
        self._axes = [joint.axes for joint in self._joints]
        self._dofs = [(u-len(self._axes[i]), u)
                      for i, u in enumerate(np.cumsum([len(joint.axes) for joint in self._joints]))]

        assert len(self._joints) == len(self.origins()), 'Wrong joints number'
        assert all([len(j.axes) == len(j.limits) for j in self._joints]), 'Wrong limits number'
        assert not self._joints[0].axes, 'Palm joint is not fixed'

    @property
    def joints(self):
        """Joint descriptions.

        Returns:
            list -- list of Joint structures
        """
        return self._joints

    @property
    def dofs_number(self):
        """Number of degrees of freedom.

        Returns:
            int -- sum of degrees of freedom of all joints
        """
        return sum([len(joint.axes) for joint in self._joints[1:]])

    @property
    def dofs_limits(self):
        """Limits corresponding to degrees of freedom.

        Returns:
            tuple -- lower limits list, upper limits list
        """
        return list(zip(*[limits for joint in self._joints[1:] for limits in joint.limits]))

    def angles_to_mano(self, angles, palm_basis=None):
        """Convert joint angles to a MANO pose.

        Arguments:
            angles {array} -- rigid model's dofs angles

        Keyword Arguments:
            palm_basis {mat3} -- palm basis (default: {None})

        Returns:
            array -- MANO pose, array of size N*3 where N - number of links
        """
        rvecs = [mat2rvec(self._basis[i] @ joint2mat(self._axes[i], angles[d0:d1]) @ self._basis[i].T)
                 for i, (d0, d1) in enumerate(self._dofs)]
        if palm_basis is not None:
            rvecs[0] = mat2rvec(palm_basis)
        return np.ravel(rvecs)

    def mano_to_angles(self, mano_pose):
        """Convert a mano pose to joint angles of the rigid model.

        It is not guaranteed that the rigid model can ideally
        recover a mano pose.

        Arguments:
            mano_pose {array} -- MANO pose, array of size N*3 where N - number of links

        Returns:
            tuple -- dofs angles, palm_basis
        """
        rvecs = np.asarray(mano_pose).reshape((-1, 3))
        angles = [angle for i, rvec in enumerate(rvecs) for angle in
                  mat2joint(self._basis[i].T @ rvec2mat(rvec) @ self._basis[i], self._axes[i])]
        palm_basis = rvec2mat(rvecs[0])
        return angles, palm_basis

    def _make_joints(self):
        """Compute joints parameters.

        Returns:
            list -- list of joints parameters
        """
        raise NotImplementedError


class HandModel20(HandModel):
    """Heuristic rigid model with 20 degrees of freedom."""

    def _make_joints(self):
        """Compute joints parameters.

        Returns:
            list -- list of joints parameters
        """
        origin = dict(zip(self.link_names, self.origins()))
        basis = {'palm': np.eye(3)}

        def make_basis(yvec, zvec):
            mat = np.vstack([np.cross(yvec, zvec), yvec, zvec])
            return mat.T / np.linalg.norm(mat.T, axis=0)

        zvec = origin['index2'] - origin['index3']
        yvec = np.cross(zvec, [0.0, 0.0, 1.0])
        basis['index'] = make_basis(yvec, zvec)

        zvec = origin['middle2'] - origin['middle3']
        yvec = np.cross(zvec, origin['index1'] - origin['ring1'])
        basis['middle'] = make_basis(yvec, zvec)

        zvec = origin['ring2'] - origin['ring3']
        yvec = np.cross(zvec, origin['middle1'] - origin['ring1'])
        basis['ring'] = make_basis(yvec, zvec)

        zvec = origin['pinky2'] - origin['pinky3']
        yvec = np.cross(zvec, origin['ring1'] - origin['pinky1'])
        basis['pinky'] = make_basis(yvec, zvec)

        yvec = origin['thumb1'] - origin['index1']
        zvec = np.cross(yvec, origin['thumb1'] - origin['thumb2'])
        basis['thumb0'] = make_basis(yvec, zvec)

        zvec = origin['thumb2'] - origin['thumb3']
        yvec = np.cross(zvec, [0, -np.sin(0.96), np.cos(0.96)])
        basis['thumb'] = make_basis(yvec, zvec)

        if self.is_left_hand:
            rot = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
            basis = {key: mat @ rot for key, mat in basis.items()}

        return [
            Joint(origin['palm'], basis['palm'], '', []),
            Joint(origin['index1'], basis['index'], 'yx', np.deg2rad([(-20, 20), (-10, 90)])),
            Joint(origin['index2'], basis['index'], 'x', np.deg2rad([(0, 100)])),
            Joint(origin['index3'], basis['index'], 'x', np.deg2rad([(0, 100)])),
            Joint(origin['middle1'], basis['middle'], 'yx', np.deg2rad([(-30, 20), (-10, 90)])),
            Joint(origin['middle2'], basis['middle'], 'x', np.deg2rad([(0, 100)])),
            Joint(origin['middle3'], basis['middle'], 'x', np.deg2rad([(0, 100)])),
            Joint(origin['pinky1'], basis['pinky'], 'yx', np.deg2rad([(-40, 20), (-10, 90)])),
            Joint(origin['pinky2'], basis['pinky'], 'x', np.deg2rad([(0, 100)])),
            Joint(origin['pinky3'], basis['pinky'], 'x', np.deg2rad([(0, 100)])),
            Joint(origin['ring1'], basis['ring'], 'yx', np.deg2rad([(-30, 20), (-10, 90)])),
            Joint(origin['ring2'], basis['ring'], 'x', np.deg2rad([(0, 100)])),
            Joint(origin['ring3'], basis['ring'], 'x', np.deg2rad([(0, 100)])),
            Joint(origin['thumb1'], basis['thumb0'], 'yz', np.deg2rad([(-10, 150), (-40, 40)])),
            Joint(origin['thumb2'], basis['thumb'], 'x', np.deg2rad([(0, 100)])),
            Joint(origin['thumb3'], basis['thumb'], 'x', np.deg2rad([(0, 100)])),
        ]


class HandModel45(HandModel):
    """Rigid model with 45 degrees of freedom."""

    def _make_joints(self):
        """Compute joints parameters.

        Returns:
            list -- list of joints parameters
        """
        origin = dict(zip(self.link_names, self.origins()))
        limits = [(-np.pi, np.pi), (-np.pi/2, np.pi/2), (-np.pi, np.pi)]

        return [
            Joint(origin['palm'], np.eye(3), '', []),
            Joint(origin['index1'], np.eye(3), 'xyz', limits),
            Joint(origin['index2'], np.eye(3), 'xyz', limits),
            Joint(origin['index3'], np.eye(3), 'xyz', limits),
            Joint(origin['middle1'], np.eye(3), 'xyz', limits),
            Joint(origin['middle2'], np.eye(3), 'xyz', limits),
            Joint(origin['middle3'], np.eye(3), 'xyz', limits),
            Joint(origin['pinky1'], np.eye(3), 'xyz', limits),
            Joint(origin['pinky2'], np.eye(3), 'xyz', limits),
            Joint(origin['pinky3'], np.eye(3), 'xyz', limits),
            Joint(origin['ring1'], np.eye(3), 'xyz', limits),
            Joint(origin['ring2'], np.eye(3), 'xyz', limits),
            Joint(origin['ring3'], np.eye(3), 'xyz', limits),
            Joint(origin['thumb1'], np.eye(3), 'xyz', limits),
            Joint(origin['thumb2'], np.eye(3), 'xyz', limits),
            Joint(origin['thumb3'], np.eye(3), 'xyz', limits),
        ]
