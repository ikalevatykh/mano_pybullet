"""Test hand_push_env module."""

import unittest

from mano_pybullet.envs.hand_push_env import HandPushEnv
from mano_pybullet.envs.wrappers.json_player import JSONPlayer


class TestHandPushEnv(unittest.TestCase):
    """Test hand_push_env module."""

    def test_trajectory_playing(self):
        env = HandPushEnv()
        env = JSONPlayer(env, './data/push_teddy.json')
        env.reset()
        done = False
        while not done:
            _observation, _reward, done, _info = env.step()
        env.close()


if __name__ == "__main__":
    unittest.main()