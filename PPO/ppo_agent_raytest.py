import wandb
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda

import gym
import numpy as np

from pathlib import Path
import ray

tf.keras.backend.set_floatx('float64')

# function for writing models out
def writeout(agents, index, title = None):
    
    Path(wandb.run.dir + "/" + "epoch-" + str(index) + "/").mkdir(parents=True, exist_ok=True)
    
    for j in range(len(agents)):
        ref = agents[j].save_weights.remote(index, wandb.run.dir, wandb.run.id, title)
        ray.get(ref)


@ray.remote
class Agent(object):

    class Actor:

        def __init__(self, state_dim, action_dim, action_bound, std_bound, config):
            self.config = config

            self.state_dim = state_dim
            self.action_dim = action_dim
            self.action_bound = action_bound
            self.std_bound = std_bound
            self.model = self.create_model()
            self.opt = tf.keras.optimizers.Adam(self.config.actor_lr)

        def get_action(self, state):
            state = np.reshape(state, [1, self.state_dim])
            mu, std = self.model.predict(state)
            action = np.random.normal(mu[0], std[0], size=self.action_dim)
            action = np.clip(action, -self.action_bound, self.action_bound)
            log_policy = self.log_pdf(mu, std, action)

            return log_policy, action

        def log_pdf(self, mu, std, action):
            std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
            var = std ** 2
            log_policy_pdf = -0.5 * (action - mu) ** 2 / \
                var - 0.5 * tf.math.log(var * 2 * np.pi)
            return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

        def create_model(self):
            state_input = Input((self.state_dim,))
            dense_1 = Dense(self.config.actor['layer1'], activation='relu')(state_input)
            dense_2 = Dense(self.config.actor['layer2'], activation='relu')(dense_1)
            out_mu = Dense(self.action_dim, activation='tanh')(dense_2)
            mu_output = Lambda(lambda x: x * self.action_bound)(out_mu)
            std_output = Dense(self.action_dim, activation='softplus')(dense_2)
            return tf.keras.models.Model(state_input, [mu_output, std_output])

        def compute_loss(self, log_old_policy, log_new_policy, actions, gaes):
            ratio = tf.exp(log_new_policy - tf.stop_gradient(log_old_policy))
            gaes = tf.stop_gradient(gaes)
            clipped_ratio = tf.clip_by_value(
                ratio, 1.0-self.config.clip_ratio, 1.0+self.config.clip_ratio)
            surrogate = -tf.minimum(ratio * gaes, clipped_ratio * gaes)
            return tf.reduce_mean(surrogate)

        def train(self, log_old_policy, states, actions, gaes):
            with tf.GradientTape() as tape:
                mu, std = self.model(states, training=True)
                log_new_policy = self.log_pdf(mu, std, actions)
                loss = self.compute_loss(
                    log_old_policy, log_new_policy, actions, gaes)
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
            return loss

    class Critic:
        def __init__(self, state_dim, config):
            self.config = config

            self.state_dim = state_dim
            self.model = self.create_model()
            self.opt = tf.keras.optimizers.Adam(self.config.critic_lr)

        def create_model(self):
            return tf.keras.Sequential([
                Input((self.state_dim,)),
                Dense(self.config.critic['layer1'], activation='relu'),
                Dense(self.config.critic['layer2'], activation='relu'),
                Dense(self.config.critic['layer3'], activation='relu'),
                Dense(1, activation='linear')
            ])

        def compute_loss(self, v_pred, td_targets):
            mse = tf.keras.losses.MeanSquaredError()
            return mse(td_targets, v_pred)

        def train(self, states, td_targets):
            with tf.GradientTape() as tape:
                v_pred = self.model(states, training=True)
                assert v_pred.shape == td_targets.shape
                loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
            return loss

    def __init__(self, config, env, iden = 0):
        self.config = config

        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]
        self.std_bound = [1e-2, 1.0]

        self.actor_opt = tf.keras.optimizers.Adam(self.config.actor_lr)
        self.critic_opt = tf.keras.optimizers.Adam(self.config.critic_lr)
        self.actor = Agent.Actor(self.state_dim, self.action_dim,
                           self.action_bound, self.std_bound, self.config)
        self.critic = Agent.Critic(self.state_dim, self.config)
        
        self.iden = iden

    def gae_target(self, rewards, v_values, next_v_value, done):
        n_step_targets = np.zeros_like(rewards)
        gae = np.zeros_like(rewards)
        gae_cumulative = 0
        forward_val = 0

        if not done:
            forward_val = next_v_value

        for k in reversed(range(0, len(rewards))):
            delta = rewards[k] + self.config.gamma * forward_val - v_values[k]
            gae_cumulative = self.config.gamma * self.config.lmbda * gae_cumulative + delta
            gae[k] = gae_cumulative
            forward_val = v_values[k]
            n_step_targets[k] = gae[k] + v_values[k]
        return gae, n_step_targets

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch

    def train(self, max_episodes=1000):
        output = []
        print("Training Agent {}".format(self.iden))

        for ep in range(max_episodes):
            state_batch = []
            action_batch = []
            reward_batch = []
            old_policy_batch = []

            episode_reward, done = 0, False

            state = self.env.reset()
            while not done:
                
                log_old_policy, action = self.actor.get_action(state)
                next_state, reward, done, _ = self.env.step(action)

                state = np.reshape(state, [1, self.state_dim])
                action = np.reshape(action, [1, self.action_dim])
                next_state = np.reshape(next_state, [1, self.state_dim])
                reward = np.reshape(reward, [1, 1])
                log_old_policy = np.reshape(log_old_policy, [1, 1])

                state_batch.append(state)
                action_batch.append(action)
                reward_batch.append((reward+8)/8)
                old_policy_batch.append(log_old_policy)

                if len(state_batch) >= self.config.update_interval or done:
                    states = self.list_to_batch(state_batch)
                    actions = self.list_to_batch(action_batch)
                    rewards = self.list_to_batch(reward_batch)
                    old_policys = self.list_to_batch(old_policy_batch)

                    v_values = self.critic.model.predict(states)
                    next_v_value = self.critic.model.predict(next_state)

                    gaes, td_targets = self.gae_target(
                        rewards, v_values, next_v_value, done)

                    for epoch in range(self.config.intervals):
                        actor_loss = self.actor.train(
                            old_policys, states, actions, gaes)
                        critic_loss = self.critic.train(states, td_targets)

                    state_batch = []
                    action_batch = []
                    reward_batch = []
                    old_policy_batch = []

                episode_reward += reward[0][0]
                state = next_state[0]
            
            print('Bot{}, EP{} EpisodeReward={}'.format(self.iden, ep, episode_reward))
            output.append(episode_reward)
            # wandb.log({'Reward' + str(self.iden): episode_reward})
        
        return output

    def evaluate(self):
        episode_reward, done = 0, False

        state = self.env.reset()
        while not done:

            action = self.actor.get_action(state) 
            action = np.clip(action, -self.action_bound, self.action_bound)

            _, action = self.actor.get_action(state)
            next_state, reward, done, _ = self.env.step(action)

            episode_reward += reward
            state = next_state

        return episode_reward

    # functions for returning things
    def save_weights(self, index, dir, id, title = None):

        mark = title
        if title == None:
            mark = self.iden

        self.actor.model.save_weights(dir + "/" + "epoch-" + str(index) + "/" + id + "-agent{}-actor".format(mark), save_format="h5")
        self.critic.model.save_weights(dir + "/" + "epoch-" + str(index) + "/" + id + "-agent{}-critic".format(mark), save_format="h5")
    
    def actor_get_weights(self):
        return self.actor.model.get_weights()

    def critic_get_weights(self):
        return self.critic.model.get_weights()

    def iden_get(self):
        return self.iden

    # function for setting things
    def actor_set_weights(self, avg):
        self.actor.model.set_weights(avg)
        return

    def critic_set_weights(self, avg):
        self.critic.model.set_weights(avg)
        return
