"""Test kinematics module."""

import unittest

import numpy as np
from numpy.random import RandomState

from mano_pybullet.mano_model import ManoModel


class TestManoModel(unittest.TestCase):
    """Test kinematics module."""

    def setUp(self):
        self.random = RandomState(7)

    def test_constructor(self):
        for left_hand in False, True:
            with self.subTest(f"ManoModel(left_hand={left_hand})"):
                mano_model = ManoModel(left_hand)
                self.assertEqual(mano_model.is_left_hand, left_hand)
                self.assertIsNotNone(mano_model.faces)
                self.assertIsNotNone(mano_model.weights)
                self.assertIsNotNone(mano_model.kintree_table)
                self.assertIsNotNone(mano_model.shapedirs)
                self.assertIsNotNone(mano_model.posedirs)
                self.assertIsNotNone(mano_model.origins())
                self.assertIsNotNone(mano_model.vertices())
                self.assertEqual(len(mano_model.link_names), len(mano_model.origins()))
                self.assertEqual(len(mano_model.tip_links), 5)

    def test_origins(self):
        """Test the MANO joints transformation."""
        mano_model = ManoModel()
        origins = mano_model.vertices(
            # pose=self.random.uniform((16, 3)),
            trans=self.random.random((3,)))
        self.assertIsNotNone(origins)

    def test_vertices(self):
        """Test the MANO vertices transformation."""
        mano_model = ManoModel()
        vertices = mano_model.vertices(
            betas=self.random.random(10) * 0.1,
            pose=self.random.random((16, 3)),
            trans=self.random.random(3))
        self.assertIsNotNone(vertices)


if __name__ == "__main__":
    unittest.main()
