"""Lift an object environment."""

from .hand_object_env import HandObjectEnv

__all__ = ('HandLiftEnv')


class HandLiftEnv(HandObjectEnv):
    """Lift an object environment class. """

    def __init__(self, target_height=0.3, **kwargs):
        """Constructor of a HandLiftEnv.

        Keyword Arguments:
            target_height {float} -- target lift height (default: {0.3})
        """
        super().__init__(self, **kwargs)
        self._target_height = target_height

    def reset(self, **kwargs):
        """Resets the environment to an initial state and returns an initial observation.

        Returns:
            observation (object): the initial observation.
        """
        # pylint: disable=arguments-differ
        observation = super().reset(**kwargs)

        # debug target marker
        self._client.addUserDebugLine(
            (0.1, 0.1, self._target_height),
            (-0.1, -0.1, self._target_height),
            (1.0, 0.0, 0.0), 2)
        self._client.addUserDebugLine(
            (0.1, -0.1, self._target_height),
            (-0.1, 0.1, self._target_height),
            (1.0, 0.0, 0.0), 4)

        return observation

    def _get_reward(self, observation):
        """Compute reward at the current environment state.

        Returns:
            observation (object): agent's observation of the current environment
        """
        _hands_state, (object_pos, _object_orn) = observation
        return object_pos[2]

    def _is_done(self, observation):
        """Check if the current environment state is final.

        Returns:
            observation (object): agent's observation of the current environment
        """
        _hands_state, (object_pos, _object_orn) = observation
        return object_pos[2] > self._target_height


if __name__ == '__main__':
    from .wrappers.json_player import JSONPlayer

    env = HandLiftEnv()
    env.show_window(True)
    env = JSONPlayer(env, './data/lift_duck.json')
    env.reset()
    while True:
        _observation, _reward, done, _info = env.step()
        if done:
            break
