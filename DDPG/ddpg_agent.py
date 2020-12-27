import wandb
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, concatenate

import numpy as np
import random
from collections import deque

from pathlib import Path
import ray

tf.keras.backend.set_floatx('float64')

class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)
    
    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])
    
    def sample(self):
        sample = random.sample(self.buffer, wandb.config.batch_size)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(wandb.config.batch_size, -1)
        next_states = np.array(next_states).reshape(wandb.config.batch_size, -1)
        return states, actions, rewards, next_states, done
    
    def size(self):
        return len(self.buffer)
    
    
class Actor:
    def __init__(self, state_dim, action_dim, action_bound):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(wandb.config.actor_lr)

    def create_model(self):
        return tf.keras.Sequential([
            Input((self.state_dim,)),
            Dense(wandb.config.actor['layer1'], activation='relu'),
            Dense(wandb.config.actor['layer2'], activation='relu'),
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
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(wandb.config.critic_lr)

    def create_model(self):
        state_input = Input((self.state_dim,))
        s1 = Dense(wandb.config.critic['state1'], activation='relu')(state_input)
        s2 = Dense(wandb.config.critic['state2'], activation='relu')(s1)
        action_input = Input((self.action_dim,))
        a1 = Dense(wandb.config.critic['actor1'], activation='relu')(action_input)
        c1 = concatenate([s2, a1], axis=-1)
        c2 = Dense(wandb.config.critic['cat1'], activation='relu')(c1)
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
    
@ray.remote
class Agent:
    def __init__(self, env, replay, actor, critic, target_actor, target_critic, iden = 0):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]

        # self.buffer = ReplayBuffer()
        self.buffer = replay

        # self.actor = Actor(self.state_dim, self.action_dim, self.action_bound)
        # self.critic = Critic(self.state_dim, self.action_dim)
        self.actor = actor
        self.critic = critic

        # self.target_actor = Actor(self.state_dim, self.action_dim, self.action_bound)
        # self.target_critic = Critic(self.state_dim, self.action_dim)
        self.target_actor = target_actor
        self.target_critic = target_critic

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
            t_actor_weights[i] = wandb.config.tau * actor_weights[i] + (1 - wandb.config.tau) * t_actor_weights[i]

        for i in range(len(critic_weights)):
            t_critic_weights[i] = wandb.config.tau * critic_weights[i] + (1 - wandb.config.tau) * t_critic_weights[i]
        
        self.target_actor.model.set_weights(t_actor_weights)
        self.target_critic.model.set_weights(t_critic_weights)


    def td_target(self, rewards, q_values, dones):
        targets = np.asarray(q_values)
        for i in range(q_values.shape[0]):
            if dones[i]:
                targets[i] = rewards[i]
            else:
                targets[i] = wandb.config.gamma * q_values[i]
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
        print("Training bot {}".format(self.iden))
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
                
            if self.buffer.size() >= wandb.config.batch_size and self.buffer.size() >= wandb.config.train_start:    # update the states if enough
                self.replay()                
            print('Bot{}, EP{} EpisodeReward={}'.format(self.iden, ep, episode_reward))
            wandb.log({'Reward' + str(self.iden): episode_reward})
            
        if(out != None):
            out[iden] = episode_reward
        else:
            return episode_reward

# function for writing models out
def writeout(agents, index, title = None):
    
    Path(wandb.run.dir + "/" + "epoch-" + str(index) + "/").mkdir(parents=True, exist_ok=True)
    
    for j in range(len(agents)):
        if title == None:
            name = j
        else:
            name = title

        agents[j].actor.model.save_weights(wandb.run.dir + "/" + "epoch-" + str(index) + "/" + wandb.run.id + "-agent{}-actor".format(name), save_format="h5")
        agents[j].critic.model.save_weights(wandb.run.dir + "/" + "epoch-" + str(index) + "/" + wandb.run.id + "-agent{}-critic".format(name), save_format="h5")
