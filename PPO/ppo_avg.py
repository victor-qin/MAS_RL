import wandb
import tensorflow as tf
import gym
import gym_quadrotor
import numpy as np

from pathlib import Path

from ppo_agent_raytest import Agent, writeout
import ray

tf.keras.backend.set_floatx('float64')

if __name__ == "__main__":
    
    try: wandb.finish()
    except: pass
    
    ####configurations
    group_temp = "123020-3"
    wandb.init(group=group_temp, project="rl-ppo-federated", mode="offline")
    wandb.run.name = wandb.run.id
    wandb.run.tags = [group_temp]
    wandb.run.notes ="Quadcopter attitude test, PPO, 7-bot w/ quarter layers"
    wandb.run.save()
    env_name = "Pendulum-v0"
    
    wandb.config.gamma = 0.99
    wandb.config.update_interval = 5
    wandb.config.actor_lr = 0.0005
    wandb.config.critic_lr = 0.001
    wandb.config.batch_size = 64
    wandb.config.clip_ratio = 0.1
    wandb.config.lmbda = 0.95
    wandb.config.intervals = 3
    
    wandb.config.episodes = 5
    wandb.config.num = 3
    wandb.config.epochs = 3

    wandb.config.actor = {'layer1': 16, 'layer2' : 16}
    wandb.config.critic = {'layer1': 16, 'layer2' : 16, 'layer3': 8}
    
    # print(wandb.config)
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

        temp = Agent.remote(configuration, env_t, i)
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
        
        jobs = []       
        # set the average
        for j in range(N):
            jobs.append(agents[j].actor_set_weights.remote(actor_avg))
            jobs.append(agents[j].critic_set_weights.remote(critic_avg))

        for k in range(len(jobs)):
            ray.get(jobs[k])

        rewards = []
        jobs = []
        for j in range(len(agents)):
            jobs.append(agents[j].evaluate.remote())

        for j in range(len(agents)):
            rewards.append(ray.get(jobs[j]))

        rewards = np.array(rewards)
        reward = np.average(rewards)
        print('Epoch={}\t Average reward={}'.format(z, reward))
        wandb.log({'batch': z, 'Epoch-avg': reward})

        if z % 50 == 0:
            writeout([agents[0]], z, "average")
            
    writeout(agents, wandb.config.epochs)

    # wrtie things out
#     for j in range(N):
#         agents[j].actor.model.save_weights(wandb.run.dir + "/" + wandb.run.id + "-agent{}-actor".format(j))        
#         agents[j].critic.model.save_weights(wandb.run.dir + "/" + wandb.run.id + "-agent{}-critic".format(j))
    
    wandb.finish()
