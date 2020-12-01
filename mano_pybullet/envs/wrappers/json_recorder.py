"""Dataset recording helper wrapper."""

import json

import gym
import numpy as np

__all__ = ('JSONRecorder')


class JSONRecorder(gym.Wrapper):
    """Wraps the environment to allow recording actions and observations to file."""

    def __init__(self, env, filename_template, use_seed=False):
        """Constructor for a Recorder.

        Arguments:
            env {gym.Env} -- environment to wrap.
            filename_template {str} -- JSON filename template (like 'record_{:04d}.json').

        Keyword Arguments:
            use_seed {bool} -- use seed as file id (substitute in the template) (default: {False}).
        """
        super().__init__(env)
        self._filename_template = filename_template
        self._use_seed = use_seed
        self._seed = None
        self._kwargs = {}
        self._observations = []
        self._actions = []
        self._counter = 0

    def reset(self, **kwargs):
        """Resets the environment to an initial state.

        Dump recorded data to file, cleanup records, save reset arguments.
        """
        if self._observations:
            self._dump()
        self._kwargs = kwargs
        observation = self.env.reset(**kwargs)
        self._observations.append(observation)
        return observation

    def step(self, action):
        """Run one timestep of the environment's dynamics.

        Record an action and observation.
        """
        self._actions.append(action)
        observation, reward, done, info = self.env.step(action)
        self._observations.append(observation)
        return observation, reward, done, info

    def close(self):
        """Close the environment.

        Dump recorded data to file.
        """
        if self._observations:
            self._dump()
        return self.env.close()

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        Record the seed value.
        """
        self._seed = seed
        return self.env.seed(seed)

    def _dump(self):
        uid = self._seed if self._use_seed else self._counter
        filename = self._filename_template.format(uid)
        data = dict(
            version=1.0,
            kwargs=self._kwargs,
            seed=self._seed,
            observations=self.env.observation_space.to_jsonable(self._observations),
            actions=self.env.action_space.to_jsonable(self._actions),
        )
        with open(filename, 'w') as json_file:
            json.dump(data, json_file, cls=_SafeJSONEncoder)
        self._counter += 1
        self._observations = []
        self._actions = []


class _SafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
