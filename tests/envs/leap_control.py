"""GUI joint control test application."""

import math
import os
import pathlib
import time

import numpy as np
import pybullet as pb
from pyleap import Controller, HandModel, Listener
from transforms3d.axangles import axangle2mat, mat2axangle

from ..math_utils import mat2pb, pb2mat
from .hand_object_env import HandObjectEnv
from .hand_lift_env import HandLiftEnv
from .hand_push_env import HandPushEnv
from .wrappers.json_recorder import JSONRecorder


class LeapListener(Listener):
    """Leap listener."""

    def __init__(self):
        super().__init__()
        self._model = HandModel()
        self._hands = []

    def on_frame(self, controller) -> None:
        frame = controller.frame()
        self._hands = frame.hands

    def get_state(self):

        def params(hand):
            frames = self._model.frames(hand)
            angles = self._model.angles(hand)

            base_frame = frames.get('wrist')

            if hand.is_left:
                angles = {k: (-r, -p, -y) for k, (r, p, y) in angles.items()}

            joint_angles = [
                -angles['index1'][1],
                -angles['index1'][0],
                -angles['index2'][0],
                -angles['index3'][0],

                -angles['middle1'][1],
                -angles['middle1'][0],
                -angles['middle2'][0],
                -angles['middle3'][0],

                -angles['pinky1'][1],
                -angles['pinky1'][0],
                -angles['pinky2'][0],
                -angles['pinky3'][0],

                -angles['ring1'][1],
                -angles['ring1'][0],
                -angles['ring2'][0],
                -angles['ring3'][0],

                -angles['thumb1'][0] + 0.3,
                angles['thumb1'][1],
                -angles['thumb2'][0] + 0.1,
                -angles['thumb3'][0] + 0.1,
            ]

            if hand.is_left:
                joint_angles[17] = -joint_angles[17]

            return (base_frame, joint_angles)

        hands = sorted(self._hands, key=lambda h: not h.is_left)
        return [params(hand) for hand in hands]


def main():
    """Leap control application."""
    import time

    M = axangle2mat([0.0, 1.0, 0.0], -np.deg2rad(85.0))
    rot = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    M2 = M @ rot
    R = np.eye(4)

    pos_scale = [1.2, 1.2, 2.0]
    pos_offset = [0, 0, -0.35]

    controller = Controller()
    listener = LeapListener()
    controller.add_listener(listener)

    env = HandObjectEnv(hand_side=HandPushEnv.HAND_BOTH)
    env.show_window(True)
    env = JSONRecorder(env, './data/tea_time_6.json')

    def get_state(hand, i):
        frame, angles = hand
        frame = frame @ R
        pos = frame[:3, 3] * pos_scale + pos_offset
        mat = frame[:3, :3]
        if i == 1:
            orn = mat2pb(mat @ M)
        else:
            orn = mat2pb(mat @ M2)
        return pos, orn, angles

    counter = 0
    while True:
        leap_state = listener.get_state()
        if len(leap_state) == 2:
            counter += 1
        if counter > 10:
            break
        time.sleep(0.1)

    state = [get_state(s, i) for i, s in enumerate(leap_state)]

    env.reset(
        initial_hands_state=state,
        model_path='$YCB_MODELS_DIR/019_pitcher_base/textured_simple.obj',
        up_axis='z')

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

    for i in range(60*5):
        leap_state = listener.get_state()
        if len(leap_state) == 2:
            action = [get_state(s, i) for i, s in enumerate(leap_state)]
            _observation, _reward, done, _info = env.step(action)
            if done:
                break
        time.sleep(1/60.0)

    env.close()


if __name__ == '__main__':
    main()
