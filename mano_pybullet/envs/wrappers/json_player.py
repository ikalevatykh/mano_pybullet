"""Dataset playing helper wrapper."""

import glob
import json
import os

import gym

__all__ = ('JSONPlayer')


class JSONPlayer(gym.Wrapper):
    """Wraps the environment to allow play actions from file."""

    def __init__(self, env, path):
        """Constructor for a Player.

        Arguments:
            env {gym.Env} -- environment to wrap.
            path {str} -- path to the recorded JSON file or to the directory with files.
        """
        super().__init__(env)
        if os.path.isdir(path):
            self._filenames = list(glob.glob(os.path.join(path, '*.json')))
        else:
            self._filenames = [path]
        self._action_iter = None
        self._counter = 0

    @property
    def records(self):
        """Number of records in the dataset.

        Returns:
            int -- files count in the dataset
        """
        return len(self._filenames)

    def reset(self, **kwargs_override):
        """Resets the environment to an initial state.

        Load recorded data from file, apply recorded arguments and seed.
        """
        seed, kwargs = self._load()
        if seed is not None:
            self.seed(seed)
        kwargs.update(kwargs_override)
        return self.env.reset(**kwargs)

    def step(self, action=None):
        """Run one timestep of the environment's dynamics.

        Load action from file.
        """
        action = next(self._action_iter, None)
        if action is None:
            raise RuntimeError('Environment was not completed during playback.')
        return self.env.step(action)

    def _load(self):
        filename = self._filenames[self._counter]
        with open(filename, 'r') as json_file:
            data = json.load(json_file)
        self._action_iter = iter(self.env.action_space.from_jsonable(data.get('actions')))
        self._counter += 1
        return data.get('seed'), data.get('kwargs')
