import tensorflow as tf
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

def create_model():
    state_input = Input((self.state_dim,), dtype = tf.float64)
    state_err = Lambda(lambda x: x - self.target)(state_input)
    out_mu = Dense(self.action_dim, activation='linear', use_bias=False)(state_err)
    mu_output = Lambda(lambda x: x * self.action_bound)(out_mu)
    std_output = Dense(self.action_dim, activation='softplus')(state_err)

def get_action(state, action_bound):
    
    k = -np.array([0.0, 0.3, 0])

    action = np.array([np.matmul(k, state)])
    action = np.clip(action, -action_bound, action_bound)
    print(action)

    return action


if __name__ == "__main__":

    gym.envs.register(
        id='Pendulum-v1',
        entry_point='pendulum_v1:PendulumEquilEnv',
        max_episode_steps=200
    ) 

    env = gym.make('Pendulum-v1')

    
    # evaluation step
    action_bound = env.action_space.high[0]
    render = False
    episode_reward, done = 0, False

    state = env.reset()
    while not done:
        if(render):
            env.render()

        action = get_action(state, action_bound)
        print('action', action)
        # action = np.clip(action, -action_bound, action_bound)
        next_state, reward, done, _ = env.step(action)
        print('state', next_state)

        episode_reward += reward
        state = next_state

    print("episode reward", episode_reward)
    print(action_bound)