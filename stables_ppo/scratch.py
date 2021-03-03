import gym

registered_envs = set(gym.envs.registry.env_specs.keys())
print('gym_quad-v0' in registered_envs)