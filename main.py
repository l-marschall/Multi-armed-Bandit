"""
 1. Bernoulli bandit problem with the following agents/policies:
a) ε-Greedy (see Sutton & Barto, 1998, chapter 2)
b) Softmax (see Sutton & Barto, 1998, chapter 2)
c) UCB1 (see Auer 2002, pg 237)
d) Thompson sampling (see Shahriari etal 2016, pg 152)

To run the main function, simply store all python scripts in one folder and run main.py!
"""
import numpy as np
import matplotlib.pyplot as plt
from environment import Environment
from bandit import MultiArmedBandit
from agent import Agent, BetaAgent
from policy import GreedyPolicy, SoftMaxPolicy, UCBPolicy


def main(N_bandits=10, epsilon=0.1):
    # =========================
    # Settings
    # =========================
    alpha = 1
    tao = 0.1
    bandit_probs = np.ones(N_bandits) - 0.5
    bandit_probs[1:] = bandit_probs[1:] - epsilon  # bandit probabilities of success
    simulations = 100  # number of simulations to perform
    trials = 100000  # number of episodes per experiment
    save_fig = True  # if false -> plot, if true save as file in same directory
    save_format = ".pdf"  # ".pdf" or ".png"

    # =========================
    # Start multi-armed bandit simulation
    # ========================
    print("Running multi-armed bandits with N_bandits = {} and agent epsilon = {}".format(N_bandits, epsilon))
    reward_history_avg = np.zeros((trials, 4))  # reward history simulation-averaged
    # action_history_sum = np.zeros((trials, N_bandits))  # sum action history
    # regret simulation-averaged, 4 is the number of agents
    regret_history_avg = np.zeros((trials, 4))

    for i in range(simulations):
        bandit = MultiArmedBandit(bandit_probs)       # initialize bandits
        agents = [
            Agent(bandit, GreedyPolicy(epsilon)),        # epsilon-Greedy
            Agent(bandit, SoftMaxPolicy(tao)),          # Softmax
            Agent(bandit, UCBPolicy(alpha)),            # UCB1
            BetaAgent(bandit, GreedyPolicy(0))          # Thompson Sampling
        ]

        for a, agent in enumerate(agents):
            (action_history, reward_history, regret_history) = Environment(
                agent, bandit, trials, epsilon)  # perform experiment

            if (i + 1) % (simulations / 20) == 0:
                print("Agent = {}".format(a+1))
                print("[Experiment {}/{}]".format(i + 1, simulations))
                print("  bandit choice history = {}".format(
                    action_history + 1))
                print("  average reward = {}".format(np.sum(reward_history) / len(reward_history)))
                print("  cumulative regret = {}".format(
                    np.sum(regret_history) / len(regret_history)))
                print("")
            # Sum up experiment reward (later to be divided to represent an average)
            reward_history_avg[:, a] += reward_history
            regret_history_avg[:, a] += regret_history

        # # Sum up action history
        # for j, (a) in enumerate(action_history):
        #     action_history_sum[j][a] += 1

    reward_history_avg /= np.float(simulations)
    regret_history_avg /= np.float(simulations)

    # =========================
    # Plot regret history results
    # =========================
    plt.close()
    plt.plot(regret_history_avg)
    plt.legend(['Greedy', 'SoftMax', 'UCB1', 'TS'], loc='upper left')
    plt.xlabel("Episode number")
    plt.ylabel("Regret accumulated".format(simulations))
    plt.title("Bandit regret history averaged, number of arms = {}, epsilon = {}".format(
        N_bandits, epsilon))
    ax = plt.gca()
    ax.set_xscale("log", nonposx='clip')
    plt.xlim([1, trials])
    if save_fig:
        output_file = "regrets_" + str(N_bandits) + "_" + str(epsilon) + save_format
        plt.savefig(output_file, bbox_inches="tight")
    else:
        plt.show()


# Driver
if __name__ == "__main__":
    main(10, 0.1)
    main(100, 0.1)
    main(10, 0.02)
    main(100, 0.02)
