"""Push an object environment."""

from .hand_object_env import HandObjectEnv

__all__ = ('HandPushEnv')


class HandPushEnv(HandObjectEnv):
    """Push an object environment class."""

    def __init__(self, target_distance=0.2, **kwargs):
        """Constructor of a HandObjectEnv.

        Keyword Arguments:
            target_distance {float} -- distance to the target (default: {0.2})
        """
        super().__init__(self, **kwargs)
        self._target_distance = target_distance

    def reset(self, **kwargs):
        """Resets the environment to an initial state and returns an initial observation.

        Returns:
            observation (object): the initial observation.
        """
        observation = super().reset(**kwargs)

        # decrease the ground plane friction
        self._client.changeDynamics(
            self._table_id, -1, lateralFriction=0.04, spinningFriction=0.04)

        # debug target marker
        self._client.addUserDebugLine(
            (self._target_distance, -0.1, 0.0),
            (self._target_distance, 0.1, 0.0),
            (1.0, 0.0, 0.0), 4)

        return observation

    def _get_reward(self, observation):
        """Compute reward at the current environment state.

        Returns:
            observation (object): agent's observation of the current environment
        """
        _hands_state, (object_pos, _object_orn) = observation
        return object_pos[0]

    def _is_done(self, observation):
        """Check if the current environment state is final.

        Returns:
            observation (object): agent's observation of the current environment
        """
        _hands_state, (object_pos, _object_orn) = observation
        return object_pos[0] > self._target_distance


if __name__ == '__main__':
    from .wrappers.json_player import JSONPlayer

    env = HandPushEnv()
    env.show_window(True)
    env = JSONPlayer(env, './data/push_teddy.json')
    env.reset()
    while True:
        _observation, _reward, done, _info = env.step()
        if done:
            break
