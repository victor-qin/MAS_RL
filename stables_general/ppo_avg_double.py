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

from gym.envs.registration import registry, register, make, spec
from quadcopter_gym.gym_quad_target import GymQuadTarget

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

    ### Loading up gym environments
    register(
          id="gym_quad-v0",
          entry_point = 'quadcopter_gym.gym_quad:GymQuad',
    )

    ##### Config Stuff
    # eps0.15-critic
    group_temp = "032421-8_normalcri-maxact"
    env_id = 'gym_quad-v0'

    # possible projects: "rl-ppo-pend-v0-3", "rl-ppo-gym_quad-v0-3", "rl-ppo-gym_quad-v0-2"
    wandb.init(group=group_temp, project="rl-ppo-gym_quad-v0-2", mode="online")
    wandb.run.name = wandb.run.id
    wandb.run.notes ="Critic avg method with normal, Actor avg w/ max, 8 agents"

    # wandb.config.gamma = 0.99 
    # wandb.config.n_steps = 16384
    # wandb.config.actor_lr = 0.0003
    # wandb.config.critic_lr = 0.0003
    # wandb.config.batch_size = 64
    # wandb.config.clip_ratio = 0.2
    # wandb.config.lmbda = 0.95
    # wandb.config.intervals = 10

    # wandb.config.episodes = 3
    # wandb.config.num_agents = 4
    # wandb.config.epochs = 40
    # wandb.config.num_cpu = 2
    # wandb.config.num_eval = 10

    wandb.config.gamma = 0.9999 
    wandb.config.n_steps = 1024
    wandb.config.actor_lr = 0.00014
    wandb.config.critic_lr = 0.00014
    wandb.config.batch_size = 32
    wandb.config.clip_ratio = 0.2
    wandb.config.lmbda = 0.98
    wandb.config.intervals = 10

    wandb.config.episodes = 10
    wandb.config.num_agents = 8
    wandb.config.epochs = 50
    wandb.config.num_cpu = 1
    wandb.config.num_eval = 10


    wandb.config.actor = {'layer1': 64, 'layer2' : 64}
    wandb.config.critic = {'layer1': 64, 'layer2' : 64}

    # average is for averaging actors
    wandb.config.average = 'max'   # normal, max, softmax, relu, epsilon
    wandb.config.avg_cri = 'normal'
    wandb.config.kappa = 1      # range 1 (all avg) to 0 (no avg)
    wandb.config.epsilon = 0.15  # range from 1 to 0 (all random to never) - epsilon greedy

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

    policy_kwargs = dict(
                        net_arch=[dict(pi=[wandb.config.actor['layer1'], wandb.config.actor['layer2']], 
                                        vf=[wandb.config.critic['layer1'], wandb.config.critic['layer2']])]
                    )
    print(policy_kwargs)

    agents = []   
    # create a bunch of agents
    for _ in range(wandb.config.num_agents):

        env = gym.make(env_id)
        # env = make_vec_env(env_id, n_envs=wandb.config.num_cpu)

        model = PPO(MlpPolicy, env, verbose=0,
                    learning_rate = wandb.config.actor_lr,
                    n_steps = wandb.config.n_steps,
                    batch_size = wandb.config.batch_size,
                    n_epochs = wandb.config.intervals,
                    gamma = wandb.config.gamma,
                    gae_lambda = wandb.config.lmbda,
                    clip_range = wandb.config.clip_ratio,
                    )
        # model = PPO(MlpPolicy, env, verbose=0,
        #              n_steps= int(wandb.config.n_steps / wandb.config.num_cpu),
        #              policy_kwargs = policy_kwargs)
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
            mean_reward, std_reward = evaluate_policy(agents[i], eval_env, n_eval_episodes=wandb.config.num_eval)
            reward_epoch.append(mean_reward)

            agents[i].save(wandb.run.dir + '/models/'+ str(z) +'-agent_ ' + str(i) +'/')
            print(f"***mean_reward for agent {i}:{mean_reward:.2f} +/- {std_reward:.2f}")
            wandb.log({'batch': z, 'steps_' + str(i): (z+1) * wandb.config.n_steps * wandb.config.episodes, 'Reward' + str(i): mean_reward})
    
        reward_epoch = np.array(reward_epoch)
        wandb.log({'batch': z, 'steps_average': (z+1) * wandb.config.n_steps * wandb.config.episodes, 'Epoch-critic': np.average(reward_epoch)})


        all_params = dict(
            (key, value)
            for key, value in agents[0].policy.state_dict().items()
            # if ("value" in key or "shared_net" in key)
        )

        #averaging step
        # get the average - actor and critic
        if wandb.config.average == "max":
            print("max")
            weights_act = max_avg(agents, reward_epoch)
        elif wandb.config.average == "softmax":
            print("softmax")
            weights_act = softmax_avg(agents, reward_epoch)
        elif wandb.config.average == "relu":
            print("relu")
            weights_act = relu_avg(agents, reward_epoch)
        elif wandb.config.average == "epsilon":
            print("epsilon")
            weights_act = epsilon_avg(agents, reward_epoch, wandb.config.epsilon)
        elif wandb.config.average == "normal":
            print("normal")
            weights_act = normal_avg(agents)
        
        # get the weights for the critic
        if wandb.config.avg_cri == "max":
            print("max")
            weights_cri = max_avg(agents, reward_epoch)
        elif wandb.config.avg_cri == "softmax":
            print("softmax")
            weights_cri = softmax_avg(agents, reward_epoch)
        elif wandb.config.avg_cri == "relu":
            print("relu")
            weights_cri = relu_avg(agents, reward_epoch)
        elif wandb.config.avg_cri == "epsilon":
            print("epsilon")
            weights_cri = epsilon_avg(agents, reward_epoch, wandb.config.epsilon)
        elif wandb.config.avg_cri == "normal":
            print("normal")
            weights_cri = normal_avg(agents)

        # actual averaging process
        if wandb.config.average is not None:
            # do averaging  **NOTE: be careful with the if statements
            mean_params = dict(
                    (
                        name,
                        th.stack([agents[i].policy.state_dict()[name] * weights_act[i] for i in range(len(agents))]).sum(dim=0)
                    )
                    for name in all_params.keys()
                    if ("policy_net" in name or "shared_net" in name or "action_net" in name) 
                    # if ("value" in name or "shared_net" in name)
                )

            mean_params.update(dict(
                    (
                        name,
                        th.stack([agents[i].policy.state_dict()[name] * weights_cri[i] for i in range(len(agents))]).sum(dim=0)
                    )
                    for name in all_params.keys()
                    if ("value" in name or "shared_net" in name)
            ))

            print(mean_params.keys())

            # pass back the average
            for i in range(wandb.config.num_agents):

                # if kappa is actually a thing
                if wandb.config.kappa != 1:
                    temp_params = dict(
                        (
                            name,
                            agents[i].policy.state_dict()[name] * (1 - wandb.config.kappa) \
                                + mean_params[name] * wandb.config.kappa,
                        )
                        for name in mean_params.keys()
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
        wandb.log({'batch': z, 'steps_average': (z+1) * wandb.config.n_steps * wandb.config.episodes,'Epoch-average': mean_reward})

    wandb.finish()

if __name__=="__main__":
    main()