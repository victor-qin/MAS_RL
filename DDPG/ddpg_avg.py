import wandb
import gym
import numpy as np
import multiprocessing as mp
from ddpg_agent import Agent, writeout
import tensorflow as tf

tf.keras.backend.set_floatx('float64')

def runner(count):
    print(count)
    print(mp.current_process())
    return agents[count].train(wandb.config.episodes)

if __name__ == "__main__":
    
    try: wandb.finish()
    except: pass
    
    ####configurations
    group_temp = "test"
    wandb.init(group=group_temp, project="rl-ddpg-federated")
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
    
    print(wandb.config)
    
    # main run    
    N = wandb.config.num
    agents = []
    
    # set up the agent
    for i in range(N):
        env_t = gym.make(env_name)
        agents.append(Agent(env_t, i))

#         print(agents[i].actor.model.get_weights())
        
    # early write out
    writeout(agents, 0)
        
    # start the training
    for z in range(wandb.config.epochs):

        rewards = np.zeros(wandb.config.num)
        jobs = []
        # train the agent
        for j in range(len(agents)):
            print('Training Agent {}'.format(agents[j].iden))
            rewards[j] = agents[j].train(wandb.config.episodes)
    
        
#             p = mp.Process(target = agents[j].train, args=(wandb.config.episodes, rewards))
#             jobs.append(p)
#             p.start()
            
#         print('what is up')
#         for i in range(len(jobs)):
#             print(i)
#             jobs[i].join()
       
    
#         temp = np.arange(0, wandb.config.num)
#         with mp.Pool() as p:  #Use me for python 3
#             rewards=p.map(runner,temp)
    
#         print(rewards)
    
#         print('what is up')
        reward = np.average(rewards)
#         reward = reward / N
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
    wandb.finish()
    
