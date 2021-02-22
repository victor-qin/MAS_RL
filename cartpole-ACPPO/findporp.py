import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Subtract
from tensorflow.keras.constraints import max_norm
import gym
import numpy as np

# looking for good porportional controller values to tune later with

# def evaluate(self, render=False):
#     episode_reward, done = 0, False

#     state = self.env.reset()
#     while not done:
#         if(render):
#             self.env.render()

#         action = self.actor.get_action(state) 
#         action = np.clip(action, -self.action_bound, self.action_bound)

#         _, action = self.actor.get_action(state)
#         next_state, reward, done, _ = self.env.step(action)

#         episode_reward += reward
#         state = next_state

#     return episode_reward

# def create_model():
#     state_input = Input((self.state_dim,), dtype = tf.float64)
#     state_err = Lambda(lambda x: x - self.target)(state_input)
#     out_mu = Dense(self.action_dim, activation='linear', use_bias=False)(state_err)
#     mu_output = Lambda(lambda x: x * self.action_bound)(out_mu)
#     std_output = Dense(self.action_dim, activation='softplus')(state_err)

# def get_action(state, action_bound, err):
    
#     # k = -np.array([8, 0.1])
#     # i = -np.array([0.005, 0.0])

#     # cor_state = [np.arctan2(state[1], state[0]), state[2]]
#     # action = np.array([np.matmul(k, cor_state)]) 
#     # + \
#         # np.matmul(i, cor_state)])
#     action = np.random.normal(action, 0.01, size=action.shape)
#     action = np.clip(action, -action_bound, action_bound)
#     print(action)

#     return action
tf.keras.backend.set_floatx('float64')

class Actor():

    def __init__(self, state_dim, action_dim, action_bound):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.std_bound = [1e-2, 1.0]
        self.target = tf.zeros((self.state_dim,), dtype=tf.float64)

        self.model = self.create_model()


    def create_model(self):

        state_input = Input(self.state_dim, dtype = tf.float64)
        state_err = Lambda(lambda x: x - self.target)(state_input)
        mu_output = Dense(self.action_dim, activation='linear', \
            use_bias=False, name='out_mu', \
            kernel_constraint = max_norm(32))(state_err)

        std_output = Lambda(lambda x: x / 100)(state_err)
        return tf.keras.models.Model(state_input, [mu_output, std_output])

    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        # print(np.arctan2(state[1], state[0]))
        # state = np.reshape([np.arctan2(state[1], state[0]), state[2]], [1, self.state_dim-1])
        print(state)
        mu, std = self.model.predict(state)

        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        action = np.random.normal(mu[0], std, size=self.action_dim)
        # if(tf.math.is_nan(mu)):
        #     # action = np.random.normal(0, 0.01, size=self.action_dim)
        action = np.clip(action, -self.action_bound, self.action_bound)
        # log_policy = self.log_pdf(mu, std, action)

        return action

if __name__ == "__main__":

    gym.envs.register(
        id='CartPole-v2',
        entry_point='continuous_cartpole:ContinuousCartPoleEnv',
        max_episode_steps=1000
    ) 

    env = gym.make('CartPole-v2')

    
    # evaluation step
    state_dim = env.observation_space.shape[0]
    # print(env.action_space.shape)
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]
    render = True
    episode_reward, done = 0, False

    actor = Actor(state_dim, action_dim, action_bound)

    state = env.reset()
    # int_err = np.array([0, 0])
    # target = np.array([0,0])
    while not done:
        if(render):
            env.render()

        # cor_state = [np.arctan2(state[1], state[0]), state[2]]
        # err = cor_state - target
        action = actor.get_action(state)
        # print('action', action)
        # action = np.clip(action, -action_bound, action_bound)
        next_state, reward, done, _ = env.step(action)
        print('state', next_state)

        episode_reward += reward
        state = next_state

    print("episode reward", episode_reward)
    print(action_bound)


# decent control for P:
# [array([-0.06971002]), array([1.08839798]), array([1.05011236]), array([1.0230358])]