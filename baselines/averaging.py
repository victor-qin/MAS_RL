import numpy as np
import ray
import random
import wandb

def assignment(agents, weights, ISRAY=True):

    critic_avg = []
    actor_avg = []

    if(ISRAY):
        ag0 = ray.get(agents[0].actor_get_weights.remote())
    else:
        ag0 = agents[0].actor_get_weights()
    for i in range(len(ag0)):

        actor_t = weights[0] * ag0[i]

        for j in range(1, len(agents)):

            if(ISRAY):
                ref = agents[j].actor_get_weights.remote()
                actor_t = actor_t + weights[j] * ray.get(ref)[i]
            else:
                actor_t = actor_t + weights[j] * agents[j].actor_get_weights()[i]

        actor_t = actor_t
        actor_avg.append(actor_t)


    if(ISRAY):
        ag0 = ray.get(agents[0].critic_get_weights.remote())
    else:
        ag0 = agents[0].critic_get_weights()
    for i in range(len(ag0)):

        critic_t = weights[0] * ag0[i]

        for j in range(1, len(agents)):

            if(ISRAY):
                ref = agents[j].critic_get_weights.remote()
                actor_t = actor_t + weights[j] * ray.get(ref)[i]
            else:
                actor_t = actor_t + weights[j] * agents[j].critic_get_weights()[i]

        critic_t = critic_t
        critic_avg.append(critic_t)

    return critic_avg, actor_avg

def normal_avg(agents):

    critic_avg = []
    actor_avg = []

    weights = np.ones(len(agents)) * (1 / len(agents))

    return assignment(agents, weights)

def max_avg(agents, end_rewards):

    top = np.argmax(end_rewards)

    weights = np.zeros(len(agents))
    weights[top] = 1

    return assignment(agents, weights)

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

    return assignment(agents, weights)

def relu_avg(agents, end_rewards):

    avg_reward = np.average(end_rewards)
    weights = np.maximum(0, end_rewards - avg_reward)
    weights /= np.sum(weights)

    return assignment(agents, weights)


def epsilon_avg(agents, end_rewards, eps):

    if np.random.random() < eps: 
        top = np.random.choice(len(agents)) 
    else: 
        top = np.argmax(end_rewards)

    weights = np.zeros(len(agents))
    weights[top] = 1

    return assignment(agents, weights)
