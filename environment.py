import numpy as np


def Environment(agent, bandit, trials, epsilon):
    action_history = []
    reward_history = []
    regret_history = np.zeros(trials)
    for episode in range(trials):
        # Choose action from agent (from current Q estimate)
        action = agent.choose_action(bandit)
        # Pick up reward from bandit for chosen action
        reward = bandit.get_reward(action)
        # Experience a regret whenenver the first arm is not chosen
        if action != 0:
            regret = epsilon
        else:
            regret = 0
        regret_history[episode] = regret + regret_history[episode - 1]
        # Update Q action-value estimates
        agent.update_Q(reward)
        # Append to history
        action_history.append(action)
        reward_history.append(reward)
    return (np.array(action_history), np.array(reward_history), np.array(regret_history))
