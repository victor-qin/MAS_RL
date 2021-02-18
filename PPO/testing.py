
import wandb
import tensorflow as tf
import gym
import numpy as np

from pathlib import Path

from ppo_agent_raytest import Agent, writeout
from averaging import normal_avg, max_avg, softmax_avg, relu_avg
import ray
from ray.tune import register_env
import argparse
from stable_baselines3.common.env_checker import check_env

import sys
sys.path.append('../Quadcopter_SimCon/Simulation/')
sys.path.append('../')

import time

# from gym_pybullet_drones.envs.single_agent_rl.FlyThruGateAviary import FlyThruGateAviary
# from gym_pybullet_drones.utils.Logger import Logger
# from gym_pybullet_drones.utils.utils import sync

# from gym_quad import GymQuad

tf.keras.backend.set_floatx('float64')


if __name__ == "__main__":
    
    try: wandb.finish()
    except: pass
    
    ####configurations
    group_temp = "012121-1_32"
    wandb.init(group=group_temp, project="rl-ppo-federated", mode="offline")
    wandb.run.name = wandb.run.id
    wandb.run.tags = [group_temp]
    wandb.run.notes ="pendulum testing 1 bots 32/16 layers, 300 epochs"
    wandb.run.save()
    # env_name = "flythrugate-aviary-v0"
    # env_name = "Pendulum-v0"
    env_name = 'gym_quad-v0'
    wandb.init(group=group_temp, project="rl-ppo-federated", mode="offline")
    
    wandb.config.gamma = 0.99
    wandb.config.update_interval = 5
    wandb.config.actor_lr = 0.0005
    wandb.config.critic_lr = 0.001
    wandb.config.batch_size = 64
    wandb.config.clip_ratio = 0.1
    wandb.config.lmbda = 0.95
    wandb.config.intervals = 3
    
    wandb.config.episodes = 1
    wandb.config.num = 1
    wandb.config.epochs = 300

    wandb.config.actor = {'layer1': 64, 'layer2' : 64}
    wandb.config.critic = {'layer1': 64, 'layer2' : 64, 'layer3': 32}
    
    wandb.config.average = "normal"    # normal, max, softmax, relu, target
    wandb.config.kappa = 1      # range 1 (all avg) to 0 (no avg)

    wandb.run.name = wandb.run.id
    wandb.run.tags = [group_temp, "8-bot", "actor-64x2", "critic-64x2/32", "avg-softmax2", env_name]
    wandb.run.notes ="pendulum testing 8 bots 64/32 layers, 300 epochs, softmax w/ stdev"

    parser = argparse.ArgumentParser()
    parser.add_argument('--jobid', type=str, default=None)
    args = parser.parse_args()
    print("args", args.jobid)

    if(args.jobid != None):
        wandb.config.jobid = args.jobid
        print("wandb", wandb.config.jobid)

    # print(wandb.config)
    # ray.init(include_dashboard=False, ignore_reinit_error=True)
    # register_env("flythrugate-aviary-v0", lambda _: FlyThruGateAviary())
    
    # main run    
    N = wandb.config.num
    agents = []
    
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    configuration = Struct(**wandb.config.as_dict())

    gym.register(
        id="gym_quad-v0",
        entry_point = 'Quadcopter_SimCon.Simulation.gym_quad:GymQuad',
    )

    # set up the agent
    for i in range(N):
        target = np.array([0, 0, 1 + i, 0, 0, 0], dtype=np.float32)
        env_t = gym.make(env_name)
        env_t.set_target(target)

        temp = Agent(configuration, env_t, i)
        # print('hello')
        # ref = temp.iden_get.remote()

        # ray.get(ref)
        # agents.append(temp)