"""Test kinematics module."""

import itertools
import unittest

import numpy as np
import pybullet as pb
from numpy.random import RandomState
from pybullet_utils import bullet_client

from mano_pybullet.hand_body import HandBody
from mano_pybullet.hand_model import HandModel20, HandModel45
from mano_pybullet.math_utils import pb2mat


class TestHandBody(unittest.TestCase):
    """Test kinematics module."""

    def setUp(self):
        self.random = RandomState(7)

    def test_model_matching(self):
        model_type_list = [HandModel20, HandModel45]

        for model_type, left_hand in itertools.product(model_type_list, (False, True)):
            with self.subTest(f'{model_type.__name__}(left_hand={left_hand})'):
                client = bullet_client.BulletClient(pb.DIRECT)
                hand_model = model_type(left_hand)
                hand = HandBody(client, hand_model)

                body_origins, body_basises = [], []
                hand_origins, hand_basises = [], []
                for hand_index, joint_index in hand._link_mapping.items():
                    state = client.getLinkState(hand.body_id, joint_index)
                    body_origins.append(state[0])
                    hand_origins.append(hand_model.joints[hand_index].origin)
                    body_basises.append(pb2mat(state[1]))
                    hand_basises.append(hand_model.joints[hand_index].basis)
                client.disconnect()

                np.testing.assert_almost_equal(body_origins, hand_origins)
                np.testing.assert_almost_equal(body_basises, hand_basises)

    def test_self_collision(self):
        client = bullet_client.BulletClient(pb.DIRECT)
        hand_model = HandModel20()
        flags = HandBody.FLAG_DEFAULT | HandBody.FLAG_USE_SELF_COLLISION
        hand = HandBody(client, hand_model, flags=flags)

        client.stepSimulation()
        points1 = client.getContactPoints()

        hand.reset([0, 0, 0], [0, 0, 0, 1], [0.35] + [0.0] * 19)
        client.stepSimulation()
        points2 = client.getContactPoints()
        client.disconnect()

        self.assertEqual(len(points1), 0)
        self.assertGreater(len(points2), 0)

    def test_mano_state_conversion(self):
        client = bullet_client.BulletClient(pb.DIRECT)
        hand_model = HandModel45()
        hand = HandBody(client, hand_model)

        pos = self.random.uniform(-1, 1, size=3)
        orn = self.random.uniform(-1, 1, size=4)
        dofs = self.random.uniform(*hand_model.dofs_limits)
        hand.reset(pos, orn, dofs)

        trans, pose = hand.get_mano_state()
        hand.reset_from_mano(trans, pose)
        trans2, pose2 = hand.get_mano_state()
        client.disconnect()

        np.testing.assert_almost_equal(trans, trans2)
        np.testing.assert_almost_equal(pose, pose2)


if __name__ == "__main__":
    unittest.main()
