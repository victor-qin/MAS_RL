import wandb
import gym
import numpy as np
# from ddpg_agent import ReplayBuffer, Actor, Critic, Agent, writeout
# from ddpg_agent_raytest import Agent, writeout



import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, concatenate
import random
from collections import deque

from pathlib import Path
import ray
import os

tf.keras.backend.set_floatx('float64')

@ray.remote
class Agent(object):

    class ReplayBuffer:
        def __init__(self, config, capacity=20000):
            self.buffer = deque(maxlen=capacity)

            self.config = config
        
        def put(self, state, action, reward, next_state, done):
            self.buffer.append([state, action, reward, next_state, done])
        
        def sample(self):
            sample = random.sample(self.buffer, self.config.batch_size)
            states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
            states = np.array(states).reshape(self.config.batch_size, -1)
            next_states = np.array(next_states).reshape(self.config.batch_size, -1)
            return states, actions, rewards, next_states, done
        
        def size(self):
            return len(self.buffer)
        
        
    class Actor:
        def __init__(self, state_dim, action_dim, action_bound, config):
            self.config = config

            self.state_dim = state_dim
            self.action_dim = action_dim
            self.action_bound = action_bound
            self.model = self.create_model()
            self.opt = tf.keras.optimizers.Adam(self.config.actor_lr)

        def create_model(self):
            return tf.keras.Sequential([
                Input((self.state_dim,)),
                Dense(self.config.actor['layer1'], activation='relu'),
                Dense(self.config.actor['layer2'], activation='relu'),
                Dense(self.action_dim, activation='tanh'),
                Lambda(lambda x: x * self.action_bound)
            ])

        def train(self, states, q_grads):
            with tf.GradientTape() as tape:
                grads = tape.gradient(self.model(states), self.model.trainable_variables, -q_grads)
            self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        
        def predict(self, state):
            return self.model.predict(state)

        def get_action(self, state):
            state = np.reshape(state, [1, self.state_dim])
            return self.model.predict(state)[0]

    class Critic:
        def __init__(self, state_dim, action_dim, config):
            self.config = config
            
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.model = self.create_model()
            self.opt = tf.keras.optimizers.Adam(self.config.critic_lr)
          
        def create_model(self):
            state_input = Input((self.state_dim,))
            s1 = Dense(self.config.critic['state1'], activation='relu')(state_input)
            s2 = Dense(self.config.critic['state2'], activation='relu')(s1)
            action_input = Input((self.action_dim,))
            a1 = Dense(self.config.critic['actor1'], activation='relu')(action_input)
            c1 = concatenate([s2, a1], axis=-1)
            c2 = Dense(self.config.critic['cat1'], activation='relu')(c1)
            output = Dense(1, activation='linear')(c2)
            return tf.keras.Model([state_input, action_input], output)
        
        def predict(self, inputs):
            return self.model.predict(inputs)
        
        def q_grads(self, states, actions):
            actions = tf.convert_to_tensor(actions)
            with tf.GradientTape() as tape:
                tape.watch(actions)
                q_values = self.model([states, actions])
                q_values = tf.squeeze(q_values)
            return tape.gradient(q_values, actions)

        def compute_loss(self, v_pred, td_targets):
            mse = tf.keras.losses.MeanSquaredError()
            return mse(td_targets, v_pred)

        def train(self, states, actions, td_targets):
            with tf.GradientTape() as tape:
                v_pred = self.model([states, actions], training=True)
                assert v_pred.shape == td_targets.shape
                loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
            return loss

    # def __init__(self, env, replay, actor, critic, target_actor, target_critic, iden = 0):
    def __init__(self, config, env, run, iden = 0):

        self.config = config
        self.run = run

        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]

        self.buffer = Agent.ReplayBuffer(config)
        # self.buffer = replay

        self.actor = Agent.Actor(self.state_dim, self.action_dim, self.action_bound, config)
        self.critic = Agent.Critic(self.state_dim, self.action_dim, config)
        # self.actor = actor
        # self.critic = critic

        self.target_actor = Agent.Actor(self.state_dim, self.action_dim, self.action_bound, config)
        self.target_critic = Agent.Critic(self.state_dim, self.action_dim, config)
        # self.target_actor = target_actor
        # self.target_critic = target_critic

        actor_weights = self.actor.model.get_weights()
        critic_weights = self.critic.model.get_weights()
        self.target_actor.model.set_weights(actor_weights)
        self.target_critic.model.set_weights(critic_weights)
        
        self.iden = iden

    def target_update(self):
        actor_weights = self.actor.model.get_weights()
        t_actor_weights = self.target_actor.model.get_weights()
        critic_weights = self.critic.model.get_weights()
        t_critic_weights = self.target_critic.model.get_weights()

        for i in range(len(actor_weights)):
            t_actor_weights[i] = self.config.tau * actor_weights[i] + (1 - self.config.tau) * t_actor_weights[i]

        for i in range(len(critic_weights)):
            t_critic_weights[i] = self.config.tau * critic_weights[i] + (1 - self.config.tau) * t_critic_weights[i]
        
        self.target_actor.model.set_weights(t_actor_weights)
        self.target_critic.model.set_weights(t_critic_weights)


    def td_target(self, rewards, q_values, dones):
        targets = np.asarray(q_values)
        for i in range(q_values.shape[0]):
            if dones[i]:
                targets[i] = rewards[i]
            else:
                targets[i] = self.config.gamma * q_values[i]
        return targets

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch
    
    def ou_noise(self, x, rho=0.15, mu=0, dt=1e-1, sigma=0.2, dim=1):
        return x + rho * (mu-x) * dt + sigma * np.sqrt(dt) * np.random.normal(size=dim)
    
    def replay(self):
        for _ in range(10):
            states, actions, rewards, next_states, dones = self.buffer.sample()
            target_q_values = self.target_critic.predict([next_states, self.target_actor.predict(next_states)])
            td_targets = self.td_target(rewards, target_q_values, dones)
            
            self.critic.train(states, actions, td_targets)
            
            s_actions = self.actor.predict(states)
            s_grads = self.critic.q_grads(states, s_actions)
            grads = np.array(s_grads).reshape((-1, self.action_dim))
            self.actor.train(states, grads)
            self.target_update()

    def train(self, max_episodes=1000, out=None):
        rewards = []

        print("Training Agent {}".format(self.iden))
        for ep in range(max_episodes):      # train a bunch of episodes
            episode_reward, done = 0, False

            state = self.env.reset()
            bg_noise = np.zeros(self.action_dim)
            while not done:    # run till done by hitting the action that's done
#                 self.env.render()
   
                action = self.actor.get_action(state)   # pick an action, add noise, clip the action           
                noise = self.ou_noise(bg_noise, dim=self.action_dim)
                action = np.clip(action + noise, -self.action_bound, self.action_bound)

                next_state, reward, done, _ = self.env.step(action)
                self.buffer.put(state, action, (reward+8)/8, next_state, done)
                bg_noise = noise     # why does the noise wander in such a weird way
                episode_reward += reward
                state = next_state
                
            if self.buffer.size() >= self.config.batch_size and self.buffer.size() >= self.config.train_start:    # update the states if enough
                self.replay()                
            print('Bot{}, EP{} EpisodeReward={}'.format(self.iden, ep, episode_reward))
    
            rewards.append(episode_reward)
            # wandb.log({'Reward' + str(self.iden): episode_reward})
            
        if(out != None):
            out[self.iden] = rewards
        else:
            return rewards


    # functions for returning things
    def save_weights(self, index, dir, id):
        print(dir)
        self.actor.model.save_weights(dir + "/" + "epoch-" + str(index) + "/" + id + "-agent{}-actor".format(self.iden), save_format="h5")
        self.critic.model.save_weights(dir + "/" + "epoch-" + str(index) + "/" + id + "-agent{}-critic".format(self.iden), save_format="h5")

    def actor_get_weights(self):
        return self.actor.model.get_weights()

    def critic_get_weights(self):
        return self.critic.model.get_weights()

    def iden_get(self):
        return self.iden

    # function for setting things
    def actor_set_weights(self, avg):
        self.actor.model.set_weights(avg)

    def critic_set_weights(self, avg):
        self.critic.model.set_weights(avg)

# function for writing models out
def writeout(agents, index, title = None):
    
    Path(wandb.run.dir + "/" + "epoch-" + str(index) + "/").mkdir(parents=True, exist_ok=True)
    
    for j in range(len(agents)):

        ref = agents[j].save_weights.remote(index, wandb.run.dir, wandb.run.id)
        ray.get(ref)


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
        k = ray.get(ref)
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
    
