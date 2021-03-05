import gym
import numpy as np
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

import wandb
import ray
import time
from averaging import normal_avg, max_avg, softmax_avg, relu_avg, epsilon_avg

#### Function stuff
def noavg_evaluate(agents, env, timesteps = 10):
    """Return mean fitness (sum of episodic rewards) for given models
    :param env - evaluation env
    :param agents - list of models for testing
    """
    episode_rewards = []
    for i in range(len(agents)):
        model = agents[i]
        for _ in range(int(timesteps / wandb.config.num_agents) + 1):
            reward_sum = 0
            done = False
            obs = env.reset()
            while not done:
                action, _states = model.predict(obs)
                obs, reward, done, info = env.step(action)
                reward_sum += reward
            episode_rewards.append(reward_sum)
    return np.mean(episode_rewards), np.std(episode_rewards)

def main():

    ##### Config Stuff
    group_temp = "030121-4_max"
    env_id = 'Pendulum-v0'

    wandb.init(group=group_temp, project="rl-ppo-federated", mode="online")
    wandb.run.name = wandb.run.id
    wandb.run.notes ="Running baselines, going max RWA for reference, unfixed sampling"

    wandb.config.gamma = 0.99 
    wandb.config.n_steps = 16384
    wandb.config.actor_lr = 0.0003
    wandb.config.critic_lr = 0.0003
    wandb.config.batch_size = 64
    wandb.config.clip_ratio = 0.2
    wandb.config.lmbda = 0.95
    wandb.config.intervals = 10

    wandb.config.episodes = 5
    wandb.config.num_agents = 4
    wandb.config.epochs = 40
    wandb.config.num_cpu = 2

    wandb.config.actor = {'layer1': 64, 'layer2' : 64}
    wandb.config.critic = {'layer1': 64, 'layer2' : 64}

    wandb.config.average = "max"    # normal, max, softmax, relu, epsilon
    wandb.config.kappa = 1      # range 1 (all avg) to 0 (no avg)
    wandb.config.epsilon = 0.1  # range from 1 to 0 (all random to never) - epsilon greedy

    wandb.run.tags = [group_temp, 
                        str(wandb.config.num_agents) + "-bot", 
                        "actor-64x2", 
                        "critic-64x2", 
                        "avg-" + str(wandb.config.average), 
                        env_id
                    ]

    #### Begin simulations
    # Set up agents and gyms
    eval_env = gym.make(env_id)

    # checkpoint_callback = CheckpointCallback(save_freq=wandb.config.n_steps, save_path= wandb.run.dir + '/models/' \
    #                         + str(z) +'-agent_ ' + str(i) +'/', name_prefix='rl_checkpoint_model')
    # eval_callback = EvalCallback(eval_env, best_model_save_path= wandb.run.dir + '/models/',
    #                             log_path= wandb.run.dir + '/logs/', eval_freq=wandb.config.n_steps,
    #                             deterministic=True, render=False)

    agents = []
    # create a bunch of agents
    for _ in range(wandb.config.num_agents):

        env = make_vec_env(env_id, n_envs=wandb.config.num_cpu)
        # env = gym.make(env_id)

        model = PPO(MlpPolicy, env, verbose=0, n_steps= wandb.config.n_steps)
        # model = PPO(MlpPolicy, env, verbose=0, n_steps= int(wandb.config.n_steps / wandb.config.num_cpu))
        agents.append(model)

    # Doing the evaluations
    # ray.init(ignore_reinit_error=True)
    # rewards_list = []
    for z in range(wandb.config.epochs):

        reward_epoch = []
        for i in range(wandb.config.num_agents):
            # learning - need a callback w/ wandb
            # Set a callback to save model every once in while
            agents[i].learn(total_timesteps = wandb.config.n_steps * wandb.config.episodes, eval_freq = 1e4)

            # epoch end eval
            mean_reward, std_reward = evaluate_policy(agents[i], eval_env, n_eval_episodes=10)
            reward_epoch.append(mean_reward)

            agents[i].save(wandb.run.dir + '/models/'+ str(z) +'-agent_ ' + str(i) +'/')
            print(f"***mean_reward for agent {i}:{mean_reward:.2f} +/- {std_reward:.2f}")
            wandb.log({'batch': z, 'runs_' + str(i): (z+1) * wandb.config.n_steps * wandb.config.episodes, 'Reward' + str(i): mean_reward})
    
        reward_epoch = np.array(reward_epoch)
        wandb.log({'batch': z, 'runs_average': (z+1)*wandb.config.num_agents * wandb.config.n_steps * wandb.config.episodes, 'Epoch-critic': np.average(reward_epoch)})

        # do averaging
        mean_params = dict(
            (key, value)
            for key, value in agents[0].policy.state_dict().items()
            # if ("policy" in key or "shared_net" in key or "action" in key)
        )

        #averaging step
        # get the average - actor and critic
        if wandb.config.average == "max":
            print("max")
            weights = max_avg(agents, reward_epoch)
        elif wandb.config.average == "softmax":
            print("softmax")
            weights = softmax_avg(agents, reward_epoch)
        elif wandb.config.average == "relu":
            print("relu")
            weights = relu_avg(agents, reward_epoch)
        elif wandb.config.average == "epsilon":
            print("epsilon")
            weights = epsilon_avg(agents, reward_epoch, wandb.config.epsilon)
        elif wandb.config.average == "normal":
            print("normal")
            weights = normal_avg(agents)
        

        if wandb.config.average is not None:
            mean_params = dict(
                    (
                        name,
                        th.stack([agents[i].policy.state_dict()[name] * weights[i] for i in range(len(agents))]).sum(dim=0)
                    )
                    for name in mean_params.keys()
                )
    
            # pass back the average
            for i in range(wandb.config.num_agents):

                # if kappa is actually a thing
                if wandb.config.kappa != 1:
                    temp_params = dict(
                        (
                            name,
                            th.stack([agents[i].policy.state_dict()[name] * (1 - wandb.config.kappa) \
                                + mean_params[name] * wandb.config.kappa \
                                for i in range(len(agents))]).sum(dim=0)
                        )
                    )
                    agents[i].policy.load_state_dict(temp_params, strict=False)

                    

                else:
                    agents[i].policy.load_state_dict(mean_params, strict=False)

            # eval average for records
            # eval average by going through all agents for records
            eval_rewards = []
            for i in range(wandb.config.num_agents):
                temp_rewards, _ = evaluate_policy(agents[i], eval_env,
                                                    n_eval_episodes= int(10 / wandb.config.num_agents) + 1, 
                                                    return_episode_rewards = True
                                                )
                eval_rewards.append(temp_rewards)
            
            eval_rewards = np.array(eval_rewards)
            mean_reward = np.mean(eval_rewards)
            std_reward = np.std(eval_rewards)
            # mean_reward, std_reward = evaluate_policy(agents[0], eval_env, n_eval_episodes=10)
            
        else:
            mean_reward, std_reward = noavg_evaluate(agents, eval_env)

        agents[0].save(wandb.run.dir + '/models/averages/' + str(z) + '-avg')
        print(f":::mean_reward for average:{mean_reward:.2f} +/- {std_reward:.2f}")
        wandb.log({'batch': z, 'runs_average': (z+1)*wandb.config.num_agents * wandb.config.n_steps * wandb.config.episodes,'Epoch-average': mean_reward})

    wandb.finish()

if __name__=="__main__":
    main()