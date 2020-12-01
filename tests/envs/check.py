"""GUI joint control test application."""

import os
import time
import pathlib

import numpy as np
import pybullet as pb

from pyleap import HandModel, Listener, Controller

from transforms3d.axangles import axangle2mat, mat2axangle

from .hand_push_env import HandPushEnv
from .hand_lift_env import HandLiftEnv
from ..math_utils import mat2pb, pb2mat

from .wrappers.json_player import JSONPlayer
from .wrappers.gif_recorder import GIFRecorder


def main():
    """Leap control application."""
    folders = list(sorted(pathlib.Path('/home/ikalevat/Downloads/YCB_Video_Models/models').iterdir()))
    ds_path = pathlib.Path('/home/ikalevat/Downloads/YCB_Video_Models/trajectories/push')

    for path in folders:
        print(path)

        env = HandPushEnv()
        env.show_window(False)
        env = GIFRecorder(env, str(ds_path / f'{path.stem}_.gif'))
        env = JSONPlayer(env, str(ds_path / f'{path.stem}.json'))

        env.reset()
        while True:
            _observation, _reward, done, _info = env.step()
            if done:
                break
        env.close()


if __name__ == '__main__':
    main()
