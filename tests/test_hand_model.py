"""Test kinematics module."""

import itertools
import unittest

import numpy as np
from numpy.random import RandomState
from transforms3d.quaternions import quat2mat

from mano_pybullet.hand_model import HandModel20, HandModel45


class TestHandModel(unittest.TestCase):
    """Test kinematics module."""

    def setUp(self):
        self.random = RandomState(7)

    def test_constructor(self):
        model_type_list = [(HandModel20, 20), (HandModel45, 45)]

        for model_type, dofs_number in model_type_list:
            hand_model = model_type()
            self.assertEqual(hand_model.dofs_number, dofs_number)
            self.assertEqual(len(hand_model.dofs_limits[0]), dofs_number)
            self.assertEqual(len(hand_model.dofs_limits[1]), dofs_number)
            self.assertEqual(len(hand_model.joints), len(hand_model.origins()))

    def test_conversions(self):
        """Test mano pose <-> joint angles conversion."""
        model_type_list = [HandModel20, HandModel45]

        for model_type, left_hand in itertools.product(model_type_list, (False, True)):
            with self.subTest(f"{model_type.__name__}(left_hand={left_hand})"):
                hand_model = model_type(left_hand)

                ang1 = self.random.uniform(*hand_model.dofs_limits)
                mat1 = quat2mat(self.random.uniform(-1, 1, size=(4,)))
                pose1 = hand_model.angles_to_mano(ang1, mat1)
                ang2, mat2 = hand_model.mano_to_angles(pose1)
                np.testing.assert_almost_equal(mat1, mat2)
                np.testing.assert_almost_equal(ang1, ang2)

                pose2 = hand_model.angles_to_mano(ang2, mat2)
                np.testing.assert_almost_equal(pose1, pose2)


if __name__ == "__main__":
    unittest.main()
