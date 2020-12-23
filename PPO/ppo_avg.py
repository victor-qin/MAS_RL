import wandb
import tensorflow as tf
import gym
import numpy as np

from pathlib import Path

from ppo_agent import Agent, writeout

tf.keras.backend.set_floatx('float64')

if __name__ == "__main__":
    
    try: wandb.finish()
    except: pass
    
    ####configurations
    group_temp = "test"
    wandb.init(group=group_temp, project="rl-ppo-federated")
    wandb.run.name = wandb.run.id
    wandb.run.tags = [group_temp]
    wandb.run.notes ="PPO running on quarter complexity in the neural net,1-bot small net, 30 simul run for 400"
    wandb.run.save()
    env_name = 'Pendulum-v0'
    
    wandb.config.gamma = 0.99
    wandb.config.update_interval = 5
    wandb.config.actor_lr = 0.0005
    wandb.config.critic_lr = 0.001
    wandb.config.batch_size = 64
    wandb.config.clip_ratio = 0.1
    wandb.config.lmbda = 0.95
    wandb.config.intervals = 3
    
    wandb.config.episodes = 5
    wandb.config.num = 1
    wandb.config.epochs = 3

    wandb.config.actor = {'layer1': 16, 'layer2' : 16}
    wandb.config.critic = {'layer1': 16, 'layer2' : 16, 'layer3': 8}
    
    print(wandb.config)
    
    # main run    
    N = wandb.config.num
    agents = []
    
    # set up the agent
    for i in range(N):
        env_t = gym.make(env_name)
        agents.append(Agent(env_t, i))

    # early write out
    writeout(agents, 0)
        
    # start the training
    for z in range(wandb.config.epochs):

        reward = 0
        # train the agent
        for j in range(len(agents)):
            print('Training Agent {}'.format(agents[j].iden))
            reward += agents[j].train(wandb.config.episodes)
    
        reward = reward / N
        print('Epoch={}\t Average reward={}'.format(z, reward))
        wandb.log({'batch': z, 'Epoch': reward})


        # get the average - actor and critic
        critic_avg = []
        actor_avg = []

        for i in range(len(agents[0].actor.model.get_weights())):
            
            actor_t = agents[0].actor.model.get_weights()[i]

            for j in range(1, N):
                actor_t += agents[j].actor.model.get_weights()[i]

            actor_t = actor_t / N
            actor_avg.append(actor_t)


        for i in range(len(agents[0].critic.model.get_weights())):
            critic_t = agents[0].critic.model.get_weights()[i]

            for j in range(1, N):
                critic_t += agents[j].critic.model.get_weights()[i]

            critic_t = critic_t / N
            critic_avg.append(critic_t)


        if z % 50 == 0:
            writeout(agents, z)
            
        # set the average
        for j in range(N):
            agents[j].actor.model.set_weights(actor_avg)
            agents[j].critic.model.set_weights(critic_avg)

        if z % 50 == 0:
            writeout([agents[0]], z, "average")
            
        writeout(agents, wandb.config.epochs)

    # wrtie things out
#     for j in range(N):
#         agents[j].actor.model.save_weights(wandb.run.dir + "/" + wandb.run.id + "-agent{}-actor".format(j))        
#         agents[j].critic.model.save_weights(wandb.run.dir + "/" + wandb.run.id + "-agent{}-critic".format(j))
    
    wandb.finish()