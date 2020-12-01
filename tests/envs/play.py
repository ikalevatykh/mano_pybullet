"""GUI joint control test application."""

import os
import pathlib
import time

import pybullet as pb

from .hand_lift_env import HandLiftEnv
from .hand_push_env import HandPushEnv
from .hand_object_env import HandObjectEnv
from .wrappers.json_player import JSONPlayer
from .wrappers.gif_recorder import GIFRecorder


def main():
    """Leap control application."""
    json_path = './data/tea_time_4.json'
    gif_path = './media/tea_time.gif'

    env = HandObjectEnv(hand_side=HandObjectEnv.HAND_BOTH)
    env = GIFRecorder(env, gif_path)
    env = JSONPlayer(env, json_path)
    
    for _ in range(env.records):
        env.reset()

        # spawn the object
        body_id = env.unwrapped._load_model('$YCB_MODELS_DIR/025_mug/textured_simple.obj')
        env.unwrapped._client.changeDynamics(
            body_id, -1, linearDamping=0.5, angularDamping=0.5)

        # put the object onto the table
        orig, _ = env.unwrapped._client.getBasePositionAndOrientation(body_id)
        bmin, _ = env.unwrapped._client.getAABB(body_id, -1)
        pose = (-0.1, 0.0, orig[2]-bmin[2]), pb.getQuaternionFromEuler((0.0, 0.0, 1.57))
        env.unwrapped._client.resetBasePositionAndOrientation(body_id, *pose)

        pose = (0.1, 0.0, 0.15), pb.getQuaternionFromEuler((0.0, 0.0, -2.0))
        env.unwrapped._client.resetBasePositionAndOrientation(env.unwrapped._body_id, *pose)

        env.unwrapped._client.changeDynamics(
            env.unwrapped._body_id, -1, linearDamping=50, angularDamping=50)

        done = False
        while not done:
            _, _, done, _ = env.step()

    env.close()


if __name__ == '__main__':
    main()
