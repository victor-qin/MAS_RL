import imageio
import numpy as np

from stable_baselines3 import PPO

env_id = 'Pendulum-v0'
eval_env = gym.make(env_id)
model = PPO(MlpPolicy, eval_env, verbose=0)
model.load

images = []
obs = model.env.reset()
img = model.env.render(mode='rgb_array')
for i in range(350):
    images.append(img)
    action, _ = model.predict(obs)
    obs, _, _ ,_ = model.env.step(action)
    img = model.env.render(mode='rgb_array')

imageio.mimsave('lander_a2c.gif', [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)