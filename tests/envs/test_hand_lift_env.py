"""Test hand_lift_env module."""

import unittest

from mano_pybullet.envs.hand_lift_env import HandLiftEnv
from mano_pybullet.envs.wrappers.json_player import JSONPlayer


class TestHandLiftEnv(unittest.TestCase):
    """Test hand_lift_env module."""

    def test_trajectory_playing(self):
        env = HandLiftEnv()
        env = JSONPlayer(env, './data/lift_duck.json')
        env.reset()
        done = False
        while not done:
            _observation, _reward, done, _info = env.step()
        env.close()


if __name__ == "__main__":
    unittest.main()
