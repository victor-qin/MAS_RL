import numpy as np
import matplotlib.pyplot as plt
import time

import gym
from gym import spaces

import os
import sys
# print(os.getcwd())

parent_dir = os.path.abspath(__file__)
os.environ["PYTHONPATH"] = parent_dir + ":" + os.environ.get("PYTHONPATH", "")
# sys.path.append('../../')
# sys.path.append('../../Quadcopter_SimCon/Simulation/')

from .gym_quad import GymQuad

class GymQuadTarget(GymQuad):

    def __init__(self, isTrack = False):
        super().__init__(isTrack=isTrack)
        
        self.target_min = np.array([-5, -5, -5, -np.pi/2, -np.pi/2, -np.pi/2])
        self.target_max = np.array([5, 5, 5, np.pi/2, np.pi/2, np.pi/2])

        self.low = np.concatenate((self.pos_min, self.quat_min, self.vel_min, self.omega_min, self.target_min))
        self.high = np.concatenate((self.pos_max, self.quat_max, self.vel_max, self.omega_max, self.target_max))

        self.observation_space = spaces.Box(        # state space of 3 pos, 3 lin vel, 4 quat, 3 rot vel
            self.low, self.high, dtype=np.float64
        )   

    def step(self, action):

        next_state, reward, done, info = super().step(action)
        return np.concatenate((next_state, self.target)), reward, done, info

    # sets target and initial position (through _seed)
    def reset(self, isTrack = False):

        super().reset(isTrack=isTrack)
        return np.concatenate((self.quad.state[0:13], self.target))