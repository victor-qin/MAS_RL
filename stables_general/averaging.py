import numpy as np
import ray
import random
import wandb

#EDITED FOR STABLE BASELINES


def normal_avg(agents):

    critic_avg = []
    actor_avg = []

    weights = np.ones(len(agents)) * (1 / len(agents))

    return weights

def max_avg(agents, end_rewards):

    top = np.argmax(end_rewards)

    weights = np.zeros(len(agents))
    weights[top] = 1

    return weights

def softmax_avg(agents, end_rewards):

    avg_reward = np.average(end_rewards)
    deviation = np.sqrt(np.average(np.power((end_rewards - avg_reward), 2)))
    e_adv_r = np.exp(((end_rewards - avg_reward) / (np.max(end_rewards) - np.min(end_rewards))).astype(float))
    e_adv_s = np.exp(((end_rewards - avg_reward) / (deviation)).astype(float))

    weights_test = e_adv_r / e_adv_r.sum()
    print("range-based weighting: ", weights_test)
    wandb.log({'range-avg*n': (np.max(weights_test) - np.min(weights_test)) * len(agents)})

    weights = e_adv_s / e_adv_s.sum()
    print("standev-based weighting: ", weights)
    wandb.log({'stdev-avg*n': (np.max(weights) - np.min(weights)) * len(agents)})

    return weights

def relu_avg(agents, end_rewards):

    avg_reward = np.average(end_rewards)
    weights = np.maximum(0, end_rewards - avg_reward)
    weights /= np.sum(weights)

    return weights


def epsilon_avg(agents, end_rewards, eps):

    if np.random.random() < eps: 
        top = np.random.choice(len(agents)) 
    else: 
        top = np.argmax(end_rewards)

    weights = np.zeros(len(agents))
    weights[top] = 1

    return weights
