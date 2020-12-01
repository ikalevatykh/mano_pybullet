"""Base Hand Environment."""

import gym
import numpy as np
import pybullet as pb
import pybullet_data as pd
from numpy.random import RandomState
from pybullet_utils.bullet_client import BulletClient

from ..hand_body import HandBody
from ..hand_model import HandModel20

__all__ = ('HandEnv')


class HandEnv(gym.Env):
    """The base Hand Environment class."""

    # The environment can have both or a single hand
    HAND_RIGHT = 1
    HAND_LEFT = 2
    HAND_BOTH = HAND_LEFT + HAND_RIGHT

    # The environment can be controlled in joint or MANO-pose spaces
    MODE_JOINT = 1
    MODE_MANO = 2
    MODE_LIST = (MODE_JOINT, MODE_MANO)

    # Shape betas magnitude
    SHAPE_BETAS_MAGNITUDE = 0.1

    metadata = {
        'render.modes': ['rgb_array'],
        'video.frames_per_second': 60.0,
        'video.output_frames_per_second': 30.0,
        'step_time': 1.0 / 60.0}

    def __init__(self,
                 hand_side=HAND_RIGHT,
                 control_mode=MODE_JOINT,
                 hand_model_cls=HandModel20,
                 randomize_hand_shape=False):
        """Constructor of a HandEnv.

        Keyword Arguments:
            hand_side {int} -- hand side: left, right or both (default: {HAND_BOTH})
            control_mode {int} -- control modalities (default: {MODE_JOINT})
            hand_model_cls {type} -- hand kinematic model (default: {HandModel20})
            randomize_hand_shape {bool} -- randomize hand shape at reset (default: {True})
        """
        assert hand_side & self.HAND_BOTH, f'Wrong hand side flag: {hand_side}'
        assert control_mode in self.MODE_LIST, f'Wrong control mode flag: {control_mode}'

        self._hand_side = hand_side
        self._control_mode = control_mode
        self._randomize_hand_shape = randomize_hand_shape

        self._show_window = False
        self._random = None
        self._client = None
        self._hand_bodies = None
        self._camera_shape = (320, 240)
        self._camera_view = None
        self._camera_proj = None

        self._hand_models = []
        if self.HAND_LEFT & hand_side:
            self._hand_models.append(hand_model_cls(left_hand=True))
        if self.HAND_RIGHT & hand_side:
            self._hand_models.append(hand_model_cls(left_hand=False))

        # if control in joint space
        if self.MODE_JOINT == control_mode:
            joint_low, joint_high = map(np.float32, self._hand_models[0].dofs_limits)
            self.action_space = gym.spaces.Tuple([
                gym.spaces.Tuple((
                    gym.spaces.Box(-5.0, 5.0, shape=(3,)),  # base position x,y,z
                    gym.spaces.Box(-1.0, 1.0, shape=(4,)),  # base quaternion x,y,z,w
                    gym.spaces.Box(joint_low, joint_high),  # joint angles
                )) for _ in self._hand_models])
            self.observation_space = gym.spaces.Tuple([
                gym.spaces.Tuple((
                    gym.spaces.Box(-5.0, 5.0, shape=(3,)),  # base position x,y,z
                    gym.spaces.Box(-1.0, 1.0, shape=(4,)),  # base quaternion x,y,z,w
                    gym.spaces.Box(-10.0, 10.0, shape=(6,)),  # base constraint forces
                    gym.spaces.Box(joint_low, joint_high),  # joint angles
                    gym.spaces.Box(-3.0, 3.0, shape=(len(joint_low),)),  # joint velocities
                    gym.spaces.Box(-1.0, 1.0, shape=(len(joint_low),)),  # joint torques
                )) for _ in self._hand_models])
        # if control in MANO space
        elif self.MODE_MANO == control_mode:
            self.action_space = gym.spaces.Tuple([
                gym.spaces.Tuple((
                    gym.spaces.Box(-5.0, 5.0, shape=(3,)),  # trans
                    gym.spaces.Box(-np.pi, np.pi, shape=(16, 3)),  # pose
                )) for _ in self._hand_models])
            self.observation_space = gym.spaces.Tuple([
                gym.spaces.Tuple((
                    gym.spaces.Box(-5.0, 5.0, shape=(3,)),  # trans
                    gym.spaces.Box(-np.pi, np.pi, shape=(16, 3)),  # pose
                )) for _ in self._hand_models])

    def show_window(self, show=True):
        """Show GUI window or not.

        Keyword Arguments:
            show {bool} -- show flag (default: {True})
        """
        self._show_window = show

    def reset(self, initial_hands_state=None, **_kwargs):
        """Resets the environment to an initial state and returns an initial observation.

        Keyword Arguments:
            initial_hands_state {int} -- override the initial hands state (default: {None})

        Returns:
            observation (object): the initial observation.
        """
        if self._client is None:
            self._client = BulletClient(pb.GUI if self._show_window else pb.DIRECT)
            self._client.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
            self._setup_camera()

        self._client.resetSimulation()
        self._client.setAdditionalSearchPath(pd.getDataPath())
        self._client.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
        self._client.setGravity(0, 0, -10)
        self._client.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)

        # randomize the handshape if needed
        betas = None
        if self._randomize_hand_shape:
            betas = self._random.rand(10) * self.SHAPE_BETAS_MAGNITUDE

        # spawn the hands
        self._hand_bodies = []
        for i, model in enumerate(self._hand_models):
            hand_body = HandBody(self._client, model, shape_betas=betas)

            if initial_hands_state is not None:
                pos, orn, angles = initial_hands_state[i]
            else:
                pos_y, orn_z = (0.10, 0.0) if model.is_left_hand else (-0.10, -np.pi)
                pos, orn = (0.0, pos_y, 0.25), pb.getQuaternionFromEuler((np.pi/2, 0.0, orn_z))
                angles = np.zeros(len(hand_body.joint_indices))

            hand_body.reset(pos, orn, angles)
            hand_body.set_target(pos, orn, angles)
            self._hand_bodies.append(hand_body)

        self._client.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)
        return self._get_observation()

    def step(self, action):
        """Run one timestep of the environment's dynamics.

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains auxiliary diagnostic information
        """
        self._take_action(action)

        num_steps = int(self.metadata.get('step_time') * 240.0)
        for _ in range(num_steps):
            self._client.stepSimulation()

        observation = self._get_observation()
        reward = self._get_reward(observation)
        done = self._is_done(observation)
        info = {}
        return observation, reward, done, info

    def render(self, mode='human'):
        """Renders the environment.

        Args:
            mode (str): the mode to render with
        """
        if mode == 'rgb_array':
            data = pb.getCameraImage(*self._camera_shape, self._camera_view, self._camera_proj)
            rgba = data[2]
            return rgba[:, :, :3]
        return super().render(mode=mode)

    def close(self):
        """Perform necessary cleanup.

        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        if self._client.isConnected():
            self._client.disconnect()

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        Args:
            seed (int): provided number generators seed
        """
        self._random = RandomState(seed)
        return [seed]

    def _take_action(self, action):
        """Compute observation of the current environment.

        Args:
            action (object): an action provided by the agent
        """
        if self.MODE_JOINT == self._control_mode:
            for hand, (pos, orn, angles) in zip(self._hand_bodies, action):
                hand.set_target(pos, orn, angles)
        elif self.MODE_MANO == self._control_mode:
            for hand, (trans, pose) in zip(self._hand_bodies, action):
                hand.set_target_from_mano(trans, pose)

    def _get_observation(self):
        """Compute observation of the current environment.

        Returns:
            observation (object): agent's observation of the current environment
        """
        if self.MODE_JOINT == self._control_mode:
            return [hand.get_state() for hand in self._hand_bodies]
        if self.MODE_MANO == self._control_mode:
            return [hand.get_mano_state() for hand in self._hand_bodies]
        return None

    def _get_reward(self, observation):
        """Compute reward at the current environment state.

        Returns:
            observation (object): agent's observation of the current environment
        """
        # pylint: disable=no-self-use, unused-argument
        return 0.0

    def _is_done(self, observation):
        """Check if the current environment state is final.

        Returns:
            observation (object): agent's observation of the current environment
        """
        # pylint: disable=no-self-use, unused-argument
        return False

    def _setup_camera(self):
        """Setup virtual camera."""
        self._client.resetDebugVisualizerCamera(
            cameraDistance=0.5,
            cameraYaw=-45.0,
            cameraPitch=-40.0,
            cameraTargetPosition=[0, 0, 0.1])

        self._camera_view = self._client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.0, 0.0, 0.1],
            distance=0.5,
            yaw=-45.0,
            pitch=-40.0,
            roll=0.0,
            upAxisIndex=2)

        self._camera_proj = self._client.computeProjectionMatrixFOV(
            fov=60.0,
            aspect=self._camera_shape[0] / self._camera_shape[1],
            nearVal=0.1,
            farVal=10.0)


if __name__ == '__main__':
    import time

    env = HandEnv(HandEnv.HAND_BOTH)
    env.show_window()
    env.seed(0)
    env.reset()

    while pb.isConnected():
        time.sleep(0.1)
