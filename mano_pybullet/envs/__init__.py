"""Hand environments."""

try:
    import gym
except ModuleNotFoundError as err:
    raise RuntimeError('Package not found: gym. Use `pip install gym` to install.') from err


gym.envs.registration.register(
    id='HandLiftEnv-v1',
    entry_point='mano_pybullet.envs:HandLiftEnv',
)

gym.envs.registration.register(
    id='HandPushEnv-v1',
    entry_point='mano_pybullet.envs:HandPushEnv',
)
