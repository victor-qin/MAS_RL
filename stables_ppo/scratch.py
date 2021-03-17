import gym
import os
import torch as th
# os.chdir('../')

# print(os.path.isdir(os.getcwd() + '/Quadcopter_SimCon/'))
from quadcopter_gym.gym_quad import GymQuad

from gym.envs.registration import registry, register, make, spec

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env

register(
    id="gym_quad-v0",
    entry_point = 'quadcopter_gym.gym_quad:GymQuad',
)

register(
    id="CartPole-v2",
    entry_point = 'quadcopter_gym.continuous_cartpole:ContinuousCartPoleEnv',
)

registered_envs = set(gym.envs.registry.env_specs.keys())
print('gym_quad-v0' in registered_envs)

# build test gyms
env = gym.make('gym_quad-v0', Tf=1)
eval_env = gym.make('gym_quad-v0', Tf=1)
print(check_env(env))

#build model, initial eval
policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=[dict(pi=[256, 256], vf=[256, 256])])
model = PPO(MlpPolicy, env, verbose=0,
            learning_rate = 0.0004,
            n_steps = 512,
            batch_size = 32,
            n_epochs = 5,
            gamma = 0.98,
            gae_lambda = 0.95,
            clip_range = 0.2,
            )
mean, std = evaluate_policy(model, eval_env)
print(mean, std)

# build model and test
model.learn(100000)
mean, std = evaluate_policy(model, eval_env)
print(mean, std)

# render demo
episode_rewards, done = 0, False
obs = eval_env.reset(isTrack = True)
while not done:
    action, _states = model.predict(obs)
    obs, reward, done, _ = eval_env.step(action)
      
    # Stats
    episode_rewards += reward

print('episodes', episode_rewards)
# eval_env.render()  


episode_rewards, done = 0, False
obs = env.reset(isTrack = True)
while not done:
    action, _states = model.predict(obs)
    obs, reward, done, _ = env.step(action)
      
    # Stats
    episode_rewards += reward

print('episodes', episode_rewards)
# env.render()  