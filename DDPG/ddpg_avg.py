import wandb
import gym
import numpy as np
# from ddpg_agent import ReplayBuffer, Actor, Critic, Agent, writeout
from ddpg_agent_raytest import Agent, writeout
import tensorflow as tf
import ray
import os

tf.keras.backend.set_floatx('float64')

if __name__ == "__main__":
    
    try: wandb.finish()
    except: pass
    
    ####configurations
    group_temp = "test-ray"
    id = wandb.util.generate_id()
    wandb.init(id = id, group=group_temp, project="rl-ddpg-federated", mode="online", resume = "allow")
    active = id
    wandb.run.name = wandb.run.id
    wandb.run.tags = [group_temp]
    wandb.run.notes = "running on half node and also 5 bots, 30 groupings"
    wandb.run.save()
    env_name = 'Pendulum-v0'
    
    wandb.config.gamma = 0.99
    wandb.config.actor_lr = 0.001
    wandb.config.critic_lr = 0.0001
    wandb.config.batch_size = 64
    wandb.config.tau = 0.005
    wandb.config.train_start = 400
    wandb.config.episodes = 5
    wandb.config.num = 3
    wandb.config.epochs = 10

    wandb.config.actor = {'layer1': 128, 'layer2' : 128}
    wandb.config.critic = {'state1': 256, 'state2': 128, 'actor1': 128, 'cat1': 64}
    
    ray.init()

    # main run    
    N = wandb.config.num
    agents = []
    
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    configuration = Struct(**wandb.config.as_dict())

    # set up the agent
    for i in range(N):
        env_t = gym.make(env_name)
        
        temp = Agent.remote(configuration, env_t, active, i)
        ref = temp.iden_get.remote()
        ray.get(ref)
        agents.append(temp)

    
    # early write out
    writeout(agents, 0)
        
    # start the training
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

        rewards = np.array(rewards)
        reward = np.average(rewards[:, -1])
        print('Epoch={}\t Average reward={}'.format(z, reward))
        wandb.log({'batch': z, 'Epoch': reward})

        # get the average - actor and critic
        critic_avg = []
        actor_avg = []

        ag0 = ray.get(agents[0].actor_get_weights.remote())
        for i in range(len(ag0)):

            actor_t = ag0[i]

            for j in range(1, wandb.config.num):
                ref = agents[j].actor_get_weights.remote()
                actor_t = actor_t + ray.get(ref)[i]

            actor_t = actor_t / wandb.config.num
            actor_avg.append(actor_t)


        ag0 = ray.get(agents[0].critic_get_weights.remote())
        for i in range(len(ag0)):

            critic_t = ag0[i]

            for j in range(1, wandb.config.num):
                ref = agents[j].critic_get_weights.remote()
                critic_t = critic_t + ray.get(ref)[i]

            critic_t = critic_t / wandb.config.num
            critic_avg.append(critic_t)

        if z % 50 == 0:
            writeout(agents, z)

        # set the average
        for j in range(N):
            agents[j].actor_set_weights.remote(actor_avg)
            agents[j].critic_set_weights.remote(critic_avg)

        if z % 50 == 0:
            writeout([agents[0]], z, "average")
            
    writeout(agents, wandb.config.epochs)
    wandb.finish()
    
