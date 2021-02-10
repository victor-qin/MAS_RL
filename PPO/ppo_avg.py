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
sys.path.append('./')
sys.path.append('Quadcopter_SimCon/Simulation/')
import time

# from gym_pybullet_drones.envs.single_agent_rl.FlyThruGateAviary import FlyThruGateAviary
# from gym_pybullet_drones.utils.Logger import Logger
# from gym_pybullet_drones.utils.utils import sync

from gym_quad import GymQuad

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
    wandb.config.num = 2
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
    ray.init(include_dashboard=False, ignore_reinit_error=True)
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
        target = np.array([0, 0, 1, 0, 0, 0], dtype=np.float32)
        env_t = gym.make(env_name)
        env_t.set_target(target)

        # check_env(env_t,
        #         warn=True,
        #         skip_render_check=True
        #         )

        temp = Agent.remote(configuration, env_t, i)
        # temp = Agent.remote(configuration, env_name, i)
        ref = temp.iden_get.remote()

        # time.sleep(100)

        ray.get(ref)
        agents.append(temp)

    # early write out
    writeout(agents, 0)
    
    time.sleep(3)
    # start the training
    max_reward = -np.inf
    for z in range(wandb.config.epochs):

        rewards = []
        jobs = []
        # train the agent
        for j in range(len(agents)):
            # print('Training Agent {}'.format(agents[j].iden))
            jobs.append(agents[j].train.remote(max_episodes = wandb.config.episodes))

        for j in range(len(agents)):
            rewards.append(ray.get(jobs[j]))
            for k in range(len(rewards[j])):
                wandb.log({'Reward' + str(j): rewards[j][k]})

        rewards = np.array(rewards, dtype=object)
        reward = np.average(rewards[:, -1])

        print('Epoch={}\t Average reward={}'.format(z, reward))
        wandb.log({'batch': z, 'Epoch-critic': reward})

        # get the average - actor and critic
        if wandb.config.average == "max":
            critic_avg, actor_avg = max_avg(agents, rewards[:, -1])
        elif wandb.config.average == "softmax":
            print("softmax")
            critic_avg, actor_avg = softmax_avg(agents, rewards[:, -1])
        elif wandb.config.average == "relu":
            print("relu")
            critic_avg, actor_avg = relu_avg(agents, rewards[:, -1])
        else:
            critic_avg, actor_avg = normal_avg(agents)

        if z % 50 == 0:
            writeout(agents, z)
        
        jobs = []       
        # set the average
        for j in range(len(agents)):
            jobs.append(agents[j].actor_set_weights.remote(actor_avg, wandb.config.kappa))
            jobs.append(agents[j].critic_set_weights.remote(critic_avg, wandb.config.kappa))

        ray.wait(jobs, num_returns = 2 * len(agents), timeout=5000)

        rewards = []
        jobs = []
        for j in range(len(agents)):
            jobs.append(agents[j].evaluate.remote(render=False))

        for j in range(len(agents)):
            rewards.append(ray.get(jobs[j]))

        rewards = np.array(rewards, dtype=object)
        reward = np.average(rewards)
        print('Epoch={}\t Average reward={}'.format(z, reward))
        wandb.log({'batch': z, 'Epoch-avg': reward})

        if reward > max_reward:
            max_reward = reward
            writeout([agents[0]], z, "MAX")

        if z % 50 == 0:
            writeout([agents[0]], z, "average")
            
    writeout([agents[0]], wandb.config.epochs, "average")
    
    wandb.finish()
