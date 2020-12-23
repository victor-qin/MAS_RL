import wandb
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda

import gym

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import json
            
class Agent:
    
    def __init__(self, env, num, iden):
        
        # initial proportional controller
        # real and test (test used for gradient descent)f
        # need to be numpy vectors
        
#         self.Kp_t = self.Kp
#         self.gain_t = self.gain
        
#         self.x0 = x0
        
        self.radius = wandb.config.radius
        self.alpha = wandb.config.alpha
        self.num = num
         
        # define the environment
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]
        self.std_bound = [1e-2, 1.0]
        
        self.Kp = np.random.randn(self.state_dim)
        self.gain = np.random.randn(self.action_dim)
        self.iden = iden

        
    # Simulate the environment and get the reward out
    def simulate(self, prop, g, ep=-1):
        state_batch = []
        action_batch = []
        reward_batch = []
        old_policy_batch = []

        episode_reward, done = 0, False

        state = self.env.reset()

        # define the action taken
        def get_action(st):
            action = prop @ state + g
            return(action)
        
        while not done:
            action = get_action(state)
            next_state, reward, done, _ = self.env.step(action)

            state = np.reshape(state, [1, self.state_dim])[0]
            action = np.reshape(action, [1, 1])[0]
            next_state = np.reshape(next_state, [1, self.state_dim])[0]
            reward = np.reshape(reward, [1, 1])[0]
            
            episode_reward += reward

        if(ep >= 0):
            print('EP{} EpisodeReward={}'.format(ep, episode_reward))
            wandb.log({'Reward' + str(self.iden): episode_reward})
        return episode_reward

    
    def learn(self, episodes):
        
        for i in range(episodes):
             # pick a vector on unit sphere to move Kp in
            # vec = (np.random.rand(self.Kp.size + self.gain.size) - 0.5)
            vec = (np.random.rand(self.Kp.size + 1) - 0.5)
            vec = self.radius * vec / np.linalg.norm(vec)
            vec_k = np.reshape(vec[:self.Kp.size], self.Kp.shape)

            vec_g = vec[-1:]

    #         rand_start = self.x0#  + np.random.randn(self.x0.size) * 0.1

            # two-point gradient descent estimate
            self.Kp_t = self.Kp + vec_k
            self.gain_t = self.gain + vec_g
            err1 = self.simulate(self.Kp_t, self.gain_t, i)

            self.Kp_t = self.Kp - vec_k
            self.gain_t = self.gain - vec_g
            err2 = self.simulate(self.Kp_t, self.gain_t, i)

            z_step =  (self.Kp.size + self.gain.size / (2 * self.radius)) * (err1 - err2)
            self.Kp = self.Kp - self.alpha * z_step * vec_k
            self.gain = self.gain - self.alpha / self.num * z_step * vec_g

def writeout(K, g, z):
    path = wandb.run.dir + "/" + "epoch-" + str(z) + "/"
    filename = path + wandb.run.id + "-average-actor"
    Path(path).mkdir(parents=True, exist_ok=True)

    f = open(filename + ".json", 'w')
    var = {'reward' : rewards[-1]}
    var['Kp_avg'] = list(K)
    var['gain_avg'] = list(g)
    json.dump({"epoch-" + str(z) : var}, f)
    f.close()
    
if __name__ == "__main__":
    
    try: wandb.finish()
    except: pass
    
    # initialize wandb
    #group="120420-1_5bot"
    wandb.init(group="120520-1_5bot", project="rl-linear-federated")
    wandb.run.name = wandb.run.id
    wandb.run.notes ="Linear controller on Pendulum-v0, 5-bot config, 400 epoch, 5 ep"
    
    env_name = 'Pendulum-v0'
    env = gym.make(env_name)
    
    wandb.config.episodes = 5
    wandb.config.num = 5
    wandb.config.epochs = 400
    wandb.config.radius = 0.001
    wandb.config.alpha = 0.003

    # initialize the controller
    agents = []
    for i in range(wandb.config.num):
        ag = Agent(env, wandb.config.num, i)
        agents.append(ag)

    # do training and averaging
    rewards = []
    rewards.append(ag.simulate(ag.Kp, ag.gain)[0])
    for z in range(wandb.config.epochs):

        Kp_avg = np.zeros(agents[0].state_dim)
        gain_avg = np.zeros(agents[0].action_dim)
        for i in range(wandb.config.num):
            print('Training Agent {}'.format(agents[i].iden))
            agents[i].learn(wandb.config.episodes)
                
            Kp_avg += agents[i].Kp
            gain_avg += agents[i].gain

        # averaging step, set agents onto average
        Kp_avg /= wandb.config.num
        gain_avg /= wandb.config.num
        for i in range(wandb.config.num):
            agents[i].Kp = Kp_avg
            agents[i].gain = gain_avg

        rewards.append(agents[0].simulate(ag.Kp, ag.gain)[0])
        print('Epoch={}\t Average reward={}'.format(z, rewards[-1]))
        wandb.log({'batch': z, 'Epoch': rewards[-1]})
        
        # write the model out if you can
        if z % 50 == 0:
            writeout(Kp_avg, gain_avg, z)
            
    wandb.finish()
#     epochs = np.arange(len(rewards))
#     plt.plot(epochs, rewards)
#     plt.plot(np.unique(epochs), np.poly1d(np.polyfit(epochs, rewards, 1))(np.unique(epochs)))

#     plt.show()
