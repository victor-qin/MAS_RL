import imageio
import numpy as np
import gym

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy

env_id = 'Pendulum-v0'
eval_env = gym.make(env_id)

model = PPO.load("./render_work/10-agent_0_2hnh6np0_none")
model.env = eval_env

result, std = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(result, std)

images = []
obs = model.env.reset()
img = model.env.render(mode='rgb_array')
tot_reward = 0
for i in range(200):
    images.append(img)
    action, _ = model.predict(obs)
    obs, reward, _ ,_ = model.env.step(action)
    img = model.env.render(mode='rgb_array')
    tot_reward += reward

print(tot_reward)
# imageio.mimsave('./render_finish/10-agent_ 0_37b22o1r_epsilon01-2.gif', \
#     [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)