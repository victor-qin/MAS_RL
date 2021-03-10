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

# from trajectory import Trajectory
# from ctrl import Control
from .quad import Quadcopter
from .utils.windModel import Wind
from .utils.gymtools import makeGymFigures, gymSameAxisAnimation
from . import utils
from . import config


class GymQuad(gym.Env):

    def __init__(self, Tf = 1.0, isTrack = False):
        self.Ti = 0
        self.Ts = 0.005
        self.Tf = Tf
        self.t = self.Ti

        self.quad = Quadcopter(self.Ti)
        self.init_pos = self.quad.pos
        self.wind = Wind('None', 2.0, 90, -15)

        # gym definition stuff
        self.action_space = spaces.Box(
          low = -1,
          high = 1,
          shape = (4,),
          dtype = np.float32
        )
        
        self.action_halfrange = float((self.quad.params["maxWmotor"] - self.quad.params["minWmotor"]) / 2)
        self.action_default = self.action_halfrange + self.quad.params["minWmotor"]

        self.pos_max = np.array([5, 5, 5])
        self.pos_min = np.array([-5, -5, -5])

        self.quat_max = np.ones(4)
        self.quat_min = -np.ones(4)

        self.vel_max = np.ones(3) * np.inf
        self.vel_min = -np.ones(3) * np.inf

        self.omega_max = np.ones(3) * np.inf
        self.omega_min = -np.ones(3) * np.inf
        
        self.low = np.concatenate((self.pos_min, self.quat_min, self.vel_min, self.omega_min))
        self.high = np.concatenate((self.pos_max, self.quat_max, self.vel_max, self.omega_max))

        self.observation_space = spaces.Box(        # state space of 3 pos, 3 lin vel, 4 quat, 3 rot vel
            self.low, self.high, dtype=np.float64
        )        

        # target definition - only location and orientation at the end of timesteps
        # NOTE: this is in NED - so the z-axis points down
        # self.target = np.zeros(6, dtype=np.float64)    # must be (6,) and within observation space
        self.target = np.array([0, 0, -1, 0, 0, 0], dtype=np.float64)
        self.epsilon = 0.25

        numTimeStep = int(self.Tf/self.Ts+1)  
        self.numTimeStep = numTimeStep

        self.init_data()
        self.isTrack = isTrack

        self.pos_err_const = 1
        self.spin_err_const = 0.05
        self.vel_err_const = 0.05

    def init_data(self):
        numTimeStep = self.numTimeStep

        self.t_all          = np.array(self.t, dtype=np.float64)
        self.s_all          = np.array([self.quad.state], dtype=np.float64)
        self.pos_all        = np.array([self.quad.pos], dtype=np.float64)
        self.vel_all        = np.array([self.quad.vel], dtype=np.float64)
        self.quat_all       = np.array([self.quad.quat], dtype=np.float64)
        self.omega_all      = np.array([self.quad.omega], dtype=np.float64)
        self.euler_all      = np.array([self.quad.euler], dtype=np.float64)
        self.w_cmd_all      = np.zeros((1,self.action_space.shape[0]), dtype=np.float64)
        self.wMotor_all     = np.array([self.quad.wMotor], dtype=np.float64)
        self.thr_all        = np.array([self.quad.thr], dtype=np.float64)
        self.tor_all        = np.array([self.quad.tor], dtype=np.float64)
        self.rewards        = np.array(0.0, dtype=np.float64)

    def track(self, i, action, reward):

        self.t_all = np.append(self.t_all, self.t)
        self.s_all = np.vstack((self.s_all, self.quad.state))
        self.pos_all = np.vstack((self.pos_all, self.quad.pos))
        self.vel_all = np.vstack((self.vel_all, self.quad.vel))
        self.quat_all = np.vstack((self.quat_all, self.quad.quat))
        self.omega_all = np.vstack((self.omega_all, self.quad.omega))
        self.euler_all = np.vstack((self.euler_all, self.quad.euler))
        self.w_cmd_all = np.vstack((self.w_cmd_all, action))
        self.wMotor_all = np.vstack((self.wMotor_all, self.quad.wMotor))
        self.thr_all = np.vstack((self.thr_all, self.quad.thr))
        self.tor_all = np.vstack((self.tor_all, self.quad.tor))
        self.rewards = np.append(self.rewards, reward)

    def set_target(self, target):

        self.target = target

    # function for calculating reward based on target position
    # input: position and orientation (7,)
    # output: reward for that state, a scalar (1,)
    def _get_reward(self, state):

        # linear cost implementation for now
        pos_err = np.linalg.norm(state[0:3] - self.target[0:3])
        spin_err = np.linalg.norm(state[10:13])
        vel_err = np.linalg.norm(state[7:10])
        # YPR = utils.quatToYPR_ZYX(state[3:7])  # translate into euler
        # eul_err = np.linalg.norm(YPR[::-1] - self.target[3:6])

        done = bool(self.t >= self.Tf)

        # reward = - (self.pos_err_const * pos_err)
        reward = - (self.pos_err_const * pos_err +
                    self.spin_err_const * spin_err + 
                    self.vel_err_const * vel_err)

        return reward, done

    def _take_action(self, action):
        adjusted_act = self.action_halfrange * action + self.action_default
        self.quad.update(self.t, self.Ts, adjusted_act, self.wind)
        self.t += self.Ts

    def step(self, action):

        self._take_action(action)
        next_state = self.quad.state[0:13]

        reward, done = self._get_reward(next_state)

        info = {'answer' : 42}

        if(self.isTrack):
            self.track(int(self.t/self.Ts), action, reward)

        # print(self.t)
        return next_state, reward, done, info

    # sets target and initial position (through _seed)
    def reset(self, isTrack = False):

        self.t = self.Ti
        # self.target = target

        rand_pos = np.random.normal(np.zeros(6), scale = 0.1)
        self._seed(rand_pos, isTrack)
        self.init_data()

        return self.quad.state[0:13]
        

    def _seed(self, rand_pos, isTrack):
        self.quad.reset(self.Ti, rand_pos)
        self.init_pos = self.quad.pos
        self.isTrack = isTrack
  
    def render(self, ifsave=0):
        
        waypoints = np.stack((self.init_pos, self.target[0:3]))
        ifsave = ifsave

        makeGymFigures(self.quad.params, np.array(self.t_all), np.array(self.pos_all), np.array(self.vel_all), np.array(self.quat_all), np.array(self.omega_all), np.array(self.euler_all), np.array(self.w_cmd_all), np.array(self.wMotor_all), np.array(self.thr_all), np.array(self.tor_all), np.array(self.rewards))
        ani = gymSameAxisAnimation(np.array(self.t_all), waypoints, np.array(self.pos_all), np.array(self.quat_all), self.Ts, self.quad.params, 1, 1, ifsave)
        plt.show()

# if __name__ == '__main__':

#     env= GymQuad()