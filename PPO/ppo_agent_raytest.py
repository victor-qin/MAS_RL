import wandb
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Subtract
from tensorflow.keras.constraints import max_norm

import gym
import numpy as np

from pathlib import Path
import ray

tf.keras.backend.set_floatx('float64')

# function for writing models out
def writeout(agents, index, title = None, ISRAY=True):
    
    Path(wandb.run.dir + "/" + "epoch-" + str(index) + "/").mkdir(parents=True, exist_ok=True)
    
    for j in range(len(agents)):
        if(ISRAY):
            ref = agents[j].save_weights.remote(index, wandb.run.dir, wandb.run.id, title)
            ray.get(ref)
        else:
            agents[j].save_weights(index, wandb.run.dir, wandb.run.id, title)


@ray.remote
class Agent(object):

    class Actor:

        def __init__(self, state_dim, action_dim, action_bound, std_bound, config):
            self.config = config
            self.target = tf.zeros(state_dim - 1, dtype=tf.float64)
            self.int_err = tf.zeros((state_dim,))

            self.state_dim = state_dim
            self.action_dim = action_dim
            self.action_bound = action_bound
            self.std_bound = std_bound
            self.model = self.create_model()

            # print(self.model.summary())
            # print(self.model.get_layer(name='out_mu').get_weights())
            self.model.get_layer(name='out_mu').set_weights([-np.array([[8.0],[0.1]])])
            # print('original', self.model.get_layer(name='out_mu').get_weights())

            self.opt = tf.keras.optimizers.Adam(self.config.actor_lr)

        def get_action(self, state):
            # state = np.reshape(state[1:], [1, self.state_dim-1])
            # print(np.arctan2(state[1], state[0]))
            state = np.reshape([np.arctan2(state[1], state[0]), state[2]], [1, self.state_dim-1])
            # action, _ = self.model.predict(state)
            # state_err = state - self.target
            # # print(state_err)
            # self.int_err += state_err
            # print(self.int_err)

            # mu, std = self.model.predict([state_err, self.int_err])
            mu, std = self.model.predict(state)
            # print(mu, std)        
            std = tf.cast(0.01, dtype=tf.float64)
            if(tf.math.is_nan(mu)):
                mu = np.zeros(mu.shape)
                std = 0.01 * np.ones(std.shape)

            action = np.random.normal(mu[0], std, size=self.action_dim)
            # if(tf.math.is_nan(mu)):
            #     # action = np.random.normal(0, 0.01, size=self.action_dim)
            action = np.clip(action, -self.action_bound, self.action_bound)
            log_policy = self.log_pdf(mu, std, action)

            return log_policy, action

        def get_real_action(self, state):
            # state = np.reshape(state[1:], [1, self.state_dim-1])
            state = np.reshape([np.arctan2(state[1], state[0]), state[2]], [1, self.state_dim-1])
            action, _ = self.model.predict(state)

            action = np.clip(action, -self.action_bound, self.action_bound)
            log_policy = None
            # log_policy = self.log_pdf(mu, std, action)

            return log_policy, action

        def log_pdf(self, mu, std, action):
            std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
            var = tf.cast(std ** 2, dtype=tf.float64)
            log_policy_pdf = -0.5 * (action - mu) ** 2 / \
                var - 0.5 * tf.cast(tf.math.log(var * 2 * np.pi), dtype=tf.float64)
            return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

        def create_model(self):
    
            state_input = Input((self.state_dim-1,), dtype = tf.float64)
            state_err = Lambda(lambda x: x - self.target)(state_input)
            mu_output = Dense(self.action_dim, activation='linear', \
                use_bias=False, name='out_mu', \
                kernel_constraint = max_norm(32))(state_err)


            # int_err = Input((self.state_dim,), dtype = tf.float64)
            # state_err = Subtract(shape=(self.state_dim,))([state_input, self.target])



            # porp_gain = Dense(self.action_dim, activation='linear')(state_err)
            # int_gain = Dense(self.action_dim, activation='linear')(int_err)
            # out_mu = Lambda(lambda x: x[0] + x[1])([porp_gain, int_gain])
            # mu_output = Lambda(lambda x: x * self.action_bound)(out_mu)
            # std_output =  Lambda(lambda x: x / 10)(Dense(self.action_dim, activation='sigmoid')(state_err))
            std_output = Lambda(lambda x: x / 100)(state_err)
            # dense_1 = Dense(self.config.actor['layer1'], activation='relu')(state_input)
            # dense_2 = Dense(self.config.actor['layer2'], activation='relu')(dense_1)
            # out_mu = Dense(self.action_dim, activation='tanh')(dense_2)
            # mu_output = Lambda(lambda x: x * self.action_bound)(out_mu)
            # std_output = Dense(self.action_dim, activation='softplus')(dense_2)
            return tf.keras.models.Model(state_input, [mu_output, std_output])
        
        def compute_loss(self, log_old_policy, log_new_policy, actions, gaes):
            # print(log_new_policy)
            # print(tf.stop_gradient(log_old_policy))
            ratio = tf.exp(log_new_policy - tf.stop_gradient(log_old_policy))
            gaes = tf.stop_gradient(gaes)
            clipped_ratio = tf.clip_by_value(
                ratio, 1.0-self.config.clip_ratio, 1.0+self.config.clip_ratio)
            surrogate = -tf.minimum(ratio * gaes, clipped_ratio * gaes)
            return tf.reduce_mean(surrogate)

        def train(self, log_old_policy, states, actions, gaes):
            backup_actor = self.model.get_weights()
            print('before', self.model.get_layer(name='out_mu').get_weights())
            with tf.GradientTape() as tape:
                # state_err = state - self.target
                # # print(state_err)
                # self.int_err += state_err
                # mu, std = self.model([state_err, self.int_err], training=True)
                states = np.transpose(np.array([np.arctan2(states[:, 1], states[:, 0]), states[:, 2]]))
                mu, std = self.model(states, training=True)

                std = tf.cast(0.01, dtype=tf.float64)
                log_new_policy = self.log_pdf(mu, std, actions)
                loss = self.compute_loss(
                    log_old_policy, log_new_policy, actions, gaes)
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

            errors = tf.math.is_nan(self.model.get_layer(name='out_mu').get_weights())
            if(len(tf.where(errors))):
                print('ERROR NaN')
                self.model.set_weights(backup_actor)

            print('after', self.model.get_layer(name='out_mu').get_weights())
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
                
                self.env.render()

                log_old_policy, action = self.actor.get_action(state)
                # print('action', action)
                next_state, reward, done, _ = self.env.step(action)
                # print('next_state', next_state)

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

    def evaluate(self, render=False):
        episode_reward, done = 0, False

        state = self.env.reset()
        while not done:
            if(render):
                self.env.render()

            # action = self.actor.get_action(state) 
            # action = np.clip(action, -self.action_bound, self.action_bound)
            

            _, action = self.actor.get_real_action(state)
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
    def actor_set_weights(self, avg, kappa=1):

        if(kappa == 1):
            self.actor.model.set_weights(avg)
        else:

            actor_weights = self.actor.model.get_weights()

            for i in range(len(actor_weights)):
                actor_weights[i] = kappa * avg[i] + (1 - kappa) * actor_weights[i]

            self.actor.model.set_weights(actor_weights)

        return

    def critic_set_weights(self, avg, kappa=1):

        if(kappa == 1):
            self.critic.model.set_weights(avg)
        else:
            critic_weights = self.critic.model.get_weights()

            for i in range(len(critic_weights)):
                critic_weights[i] = kappa * avg[i] + (1 - kappa) * critic_weights[i]

            self.critic.model.set_weights(critic_weights)

        return
