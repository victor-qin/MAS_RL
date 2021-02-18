import numpy as np
import matplotlib.pyplot as plt
import time

import gym
from gym.envs.registration import registry, register, make, spec

# from trajectory import Trajectory
# from ctrl import Control
# from quadFiles.quad import Quadcopter
# from utils.windModel import Wind
import utils
import config

from gym_quad import GymQuad
import ray

@ray.remote
def runner(env):

    for i in range(int(env.Tf / env.Ts / 2)):
        action = np.array([75, 75, 75, 75])
        env.step(action)

    for i in range(int(env.Tf / env.Ts / 2)):
        action = np.array([600, 600, 600, 600])
        env.step(action)

    print(env.w_cmd_all) 

    env.render()  

def main():

    register(
        id="gym_quad-v0",
        entry_point = 'gym_quad:GymQuad',
    )

    # utils.YPRToQuat(psi0, theta0, phi0) 
    target = np.array([3, 2, 1, 0, 0, 0], dtype=np.float32)
    env = gym.make('gym_quad-v0')
    env.set_target(target)

    # env = GymQuad()
    
    env.reset(isTrack = True)

    # for i in range(int(env.Tf / env.Ts / 2)):
    #     action = np.array([75, 75, 75, 75])
    #     env.step(action)

    # for i in range(int(env.Tf / env.Ts / 2)):
    #     action = np.array([600, 600, 600, 600])
    #     env.step(action)

    ray.init()
    job = runner.remote(env)

    ray.wait([job], num_returns=1)
    ray.get(job)

    # time.sleep(1)
    # print(env.rewards)    
    # env.render()

if __name__ == '__main__':
    main()
    
