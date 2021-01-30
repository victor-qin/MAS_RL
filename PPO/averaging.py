import numpy as np
import ray


def normal_avg(agents):

    critic_avg = []
    actor_avg = []

    ag0 = ray.get(agents[0].actor_get_weights.remote())
    for i in range(len(ag0)):

        actor_t = ag0[i]

        for j in range(1, len(agents)):
            ref = agents[j].actor_get_weights.remote()
            actor_t = actor_t + ray.get(ref)[i]

        actor_t = actor_t / len(agents)
        actor_avg.append(actor_t)


    ag0 = ray.get(agents[0].critic_get_weights.remote())
    for i in range(len(ag0)):

        critic_t = ag0[i]

        for j in range(1, len(agents)):
            ref = agents[j].critic_get_weights.remote()
            critic_t = critic_t + ray.get(ref)[i]

        critic_t = critic_t / len(agents)
        critic_avg.append(critic_t)

    return critic_avg, actor_avg

def max_avg(agents, end_rewards):

    top = np.argmax(end_rewards)

    critic_avg = []
    actor_avg = []

    ag0 = ray.get(agents[top].actor_get_weights.remote())
    for i in range(len(ag0)):

        actor_t = ag0[i]
        actor_avg.append(actor_t)


    ag0 = ray.get(agents[top].critic_get_weights.remote())
    for i in range(len(ag0)):

        critic_t = ag0[i]
        critic_avg.append(critic_t)

    return critic_avg, actor_avg

def softmax_avg(agents, end_rewards):

    avg_reward = np.average(end_rewards)
    e_adv = np.exp(((end_rewards - avg_reward) / (np.max(end_rewards) - np.min(end_rewards))).astype(float))

    weights = e_adv / e_adv.sum()

    critic_avg = []
    actor_avg = []

    ag0 = ray.get(agents[0].actor_get_weights.remote())
    for i in range(len(ag0)):

        actor_t = weights[0] * ag0[i]

        for j in range(1, len(agents)):
            ref = agents[j].actor_get_weights.remote()
            actor_t = actor_t + weights[j] * ray.get(ref)[i]

        actor_t = actor_t
        actor_avg.append(actor_t)


    ag0 = ray.get(agents[0].critic_get_weights.remote())
    for i in range(len(ag0)):

        critic_t = weights[0] * ag0[i]

        for j in range(1, len(agents)):
            ref = agents[j].critic_get_weights.remote()
            critic_t = critic_t + weights[j] * ray.get(ref)[i]

        critic_t = critic_t
        critic_avg.append(critic_t)

    return critic_avg, actor_avg

def relu_avg(agents, end_rewards):

    avg_reward = np.average(end_rewards)
    weights = np.maximum(0, end_rewards - avg_reward)
    weights /= np.sum(weights)

    critic_avg = []
    actor_avg = []

    ag0 = ray.get(agents[0].actor_get_weights.remote())
    for i in range(len(ag0)):

        actor_t = weights[0] * ag0[i]

        for j in range(1, len(agents)):
            ref = agents[j].actor_get_weights.remote()
            actor_t = actor_t + weights[j] * ray.get(ref)[i]

        actor_t = actor_t
        actor_avg.append(actor_t)


    ag0 = ray.get(agents[0].critic_get_weights.remote())
    for i in range(len(ag0)):

        critic_t = weights[0] * ag0[i]

        for j in range(1, len(agents)):
            ref = agents[j].critic_get_weights.remote()
            critic_t = critic_t + weights[j] * ray.get(ref)[i]

        critic_t = critic_t
        critic_avg.append(critic_t)

    return critic_avg, actor_avg
