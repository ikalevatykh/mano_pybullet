"""Test hand_object_env module."""

import itertools
import unittest

from mano_pybullet.envs.hand_object_env import HandObjectEnv


class TestHandObjectEnv(unittest.TestCase):
    """Test hand_object_env module."""

    def test_constructor(self):
        model_path_list = ('duck_vhacd.urdf', 'teddy_vhacd.urdf')
        up_axis_list = ('x', 'y', 'z')
        args = itertools.product(model_path_list, up_axis_list)

        for model_path, up_axis in args:
            env = HandObjectEnv(model_path=model_path, up_axis=up_axis)
            env.reset()
            env.close()

    def test_reset_args(self):
        env = HandObjectEnv()
        env.reset(model_path='duck_vhacd.urdf', up_axis='z')
        env.close()


if __name__ == "__main__":
    unittest.main()
