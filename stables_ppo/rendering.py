import imageio
import numpy as np
import gym
from gym.envs.registration import registry, register, make, spec

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy

from quadcopter_gym.gym_quad import GymQuad


register(
    id="gym_quad-v0",
    entry_point = 'quadcopter_gym.gym_quad:GymQuad',
)

env_id = 'gym_quad-v0'
eval_env = gym.make(env_id)

model = PPO.load("./render_work/gym_quad-v0_47_model")
model.env = eval_env

# result, std = evaluate_policy(model, eval_env, n_eval_episodes=10)
# print(result, std)

images = []
obs = eval_env.reset(isTrack = True)
done = False
print('trying to check this')
# img = model.env.render(mode='rgb_array')
tot_reward = 0
while not done:
    # images.append(img)
    action, _ = model.predict(obs)

    obs, reward, done ,_ = eval_env.step(action)
    # img = model.env.render(mode='rgb_array')
    tot_reward += reward

print(tot_reward)
eval_env.render()

# imageio.mimsave('./render_finish/10-agent_ 0_37b22o1r_epsilon01-2.gif', \
#     [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)