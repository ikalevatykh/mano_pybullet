"""Test hand_env module."""

import itertools
import unittest

import numpy as np
import pybullet as pb
from numpy.random import RandomState

from mano_pybullet.envs.hand_env import HandEnv
from mano_pybullet.hand_model import HandModel20, HandModel45


class TestHandEnv(unittest.TestCase):
    """Test hand_env module."""

    def setUp(self):
        self.random = RandomState(7)

    def test_constructor_default(self):
        env = HandEnv()
        self.assertEqual(env._hand_side, HandEnv.HAND_RIGHT)
        self.assertEqual(env._control_mode, HandEnv.MODE_JOINT)
        self.assertIsInstance(env._hand_models[0], HandModel20)

    def test_constructor_args(self):
        hand_side_list = (HandEnv.HAND_LEFT, HandEnv.HAND_RIGHT, HandEnv.HAND_BOTH)
        control_mode_list = (HandEnv.MODE_JOINT, HandEnv.MODE_MANO)
        model_cls_list = (HandModel20, HandModel45)
        args = itertools.product(hand_side_list, control_mode_list, model_cls_list)

        for hand_side, control_mode, model_cls in args:
            env = HandEnv(hand_side=hand_side, control_mode=control_mode, hand_model_cls=model_cls)
            env.reset()
            env.close()

    def test_reset_args(self):
        env = HandEnv()
        pos = self.random.uniform(-1.0, 1.0, size=(3,))
        orn = pb.getQuaternionFromEuler(
            self.random.uniform([-np.pi, -np.pi/2, -np.pi], [np.pi, np.pi/2, np.pi]))
        angles = self.random.uniform(*env._hand_models[0].dofs_limits)
        observation = env.reset(initial_hands_state=[[pos, orn, angles]])
        env.close()
        np.testing.assert_almost_equal(observation[0][0], pos)
        np.testing.assert_almost_equal(observation[0][1], orn)
        np.testing.assert_almost_equal(observation[0][3], angles)

    def test_show_window(self):
        env = HandEnv()
        self.assertFalse(env._show_window)
        env.show_window(True)
        self.assertTrue(env._show_window)


if __name__ == "__main__":
    unittest.main()
