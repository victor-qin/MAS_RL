import wandb
import tensorflow as tf
import gym
import gym_quadrotor
import numpy as np

from pathlib import Path

from ppo_agent_raytest import Agent, writeout
import ray

tf.keras.backend.set_floatx('float64')

def save_frames_as_gif(frames, filename, path='./renders'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

if __name__ == '__main__':

    return 1