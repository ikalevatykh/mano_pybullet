"""Base Hand with an Object Environment."""

import os

import gym
import numpy as np
import pybullet as pb

from .hand_env import HandEnv

__all__ = ('HandObjectEnv')


class HandObjectEnv(HandEnv):
    """The base Hand with Object Environment class. """

    def __init__(self, model_path=None, up_axis='z', make_convex=True, **kwargs):
        """Constructor a HandObjectEnv.

        Keyword Arguments:
            model_path {str|list} -- path to an object model file(s) (default: {None})
            up_axis {str} -- object up axis (default: {'z'})
            make_convex {bool} -- generate a convex collision mesh (default: {True})
        """
        super().__init__(**kwargs)
        self._model_path = model_path
        self._up_axis = up_axis
        self._make_convex = make_convex
        self._body_id = -1
        self._table_id = -1

        self.observation_space = gym.spaces.Tuple((
            self.observation_space,
            gym.spaces.Tuple((
                gym.spaces.Box(-5.0, 5.0, shape=(3,)),  # object's base position
                gym.spaces.Box(-1.0, 1.0, shape=(4,)),  # object's base quaternion
            ))))

    def reset(self, model_path=None, up_axis=None, **kwargs):
        """Resets the environment to an initial state and returns an initial observation.

        Returns:
            observation (object): the initial observation.

        Keyword Arguments:
            model_path {str|list} -- override the path to an object model file (default: {None})
            up_axis {str} -- override the object up axis (default: {None})
        """
        up_axis = up_axis or self._up_axis
        model_path = model_path or self._model_path
        self._body_id = -1

        super().reset(**kwargs)
        self._client.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)

        # spawn the ground plane
        shape_id = self._client.createCollisionShape(pb.GEOM_PLANE)
        self._table_id = self._client.createMultiBody(0.0, shape_id)

        # spawn the object
        if isinstance(model_path, (list, tuple)):
            model_path = self._random.choice(model_path)
        self._body_id = self._load_model(model_path)
        self._client.changeDynamics(
            self._body_id, -1, linearDamping=0.5, angularDamping=0.5)

        # put the object onto the ground plane
        orig, _ = self._client.getBasePositionAndOrientation(self._body_id)
        bmin, _ = self._client.getAABB(self._body_id, -1)
        if up_axis == 'x':
            pose = (0.0, 0.0, orig[0]-bmin[0]), pb.getQuaternionFromEuler((0.0, np.pi/2, 0.0))
        elif up_axis == 'y':
            pose = (0.0, 0.0, orig[1]-bmin[1]), pb.getQuaternionFromEuler((np.pi/2, 0.0, 0.0))
        elif up_axis == 'z':
            pose = (0.0, 0.0, orig[2]-bmin[2]), (0.0, 0.0, 0.0, 1.0)
        self._client.resetBasePositionAndOrientation(self._body_id, *pose)

        self._client.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)
        return self._get_observation()

    def _load_model(self, path):
        """Load model from file.

        Arguments:
            path {str} -- path to an object model file
        """
        path = os.path.expandvars(path)
        root, ext = os.path.splitext(path)
        if ext.lower() == '.urdf':
            return self._client.loadURDF(path)
        if ext.lower() == '.obj':
            vid = self._client.createVisualShape(pb.GEOM_MESH, fileName=path)
            if self._make_convex:
                # generate the convex decomposed mesh and save it on disk
                in_path = path
                path = root + '_vhacd' + ext
                if not os.path.exists(path):
                    self._client.vhacd(in_path, path, '')
            cid = self._client.createCollisionShape(pb.GEOM_MESH, fileName=path)
            return self._client.createMultiBody(0.2, cid, vid)
        raise RuntimeError(f'Cannot load model from: "{path}"')

    def _get_observation(self):
        """Compute observation of the current environment.

        Returns:
            observation (object): agent's observation of the current environment
        """
        if self._body_id >= 0:
            hands_observation = super()._get_observation()
            base_pos, base_orn = self._client.getBasePositionAndOrientation(self._body_id)
            return hands_observation, (base_pos, base_orn)
        return None


if __name__ == '__main__':
    import time

    env = HandObjectEnv()
    env.show_window()
    env.reset(model_path='duck_vhacd.urdf', up_axis='y')

    while pb.isConnected():
        time.sleep(0.1)
