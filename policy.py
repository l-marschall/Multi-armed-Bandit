import numpy as np


class GreedyPolicy:
    """
    The Epsilon-Greedy policy will choose a random action with probability
    epsilon and take the best apparent approach with probability 1-epsilon. If
    multiple actions are tied for best choice, then a random action from that
    subset is selected.
    """

    def __init__(self, epsilon):
        self.epsilon = epsilon

    # Choose action using an epsilon-greedy agent
    def choose_action(self, bandit, agent):
        rand = np.random.random()  # [0.0,1.0]
        if rand < self.epsilon:
            action_explore = np.random.randint(bandit.N)  # explore random bandit
            return action_explore
        else:
            # action_greedy = np.argmax(self.Q)  # exploit best current bandit
            action_greedy = np.random.choice(np.flatnonzero(agent.Q == agent.Q.max()))
            return action_greedy


class SoftMaxPolicy:
    """
    The Softmax policy converts the estimated arm rewards into probabilities
    then randomly samples from the resultant distribution. This policy is
    primarily employed by the Gradient Agent for learning relative preferences.
    """

    def __init__(self, tao):
        self.tao = tao

    # Choose action using a softmax agent
    def choose_action(self, bandit, agent):
        pi = np.exp(agent.Q/self.tao) / np.sum(np.exp(agent.Q/self.tao))
        cdf = np.cumsum(pi)
        s = np.random.random()
        action = np.where(s < cdf)[0][0]
        return action


class UCBPolicy:
    """
    The Upper Confidence Bound algorithm (UCB1). It applies an exploration
    factor to the expected value of each arm which can influence a greedy
    selection strategy to more intelligently explore less confident options.
    """

    def __init__(self, alpha):
        self.alpha = alpha
        self.t = 0

    # Choose action using an UCB1 agent
    def choose_action(self, bandit, agent):
        self.t += 1
        exploration = 2 * np.log(np.sum(self.t)) / agent.k
        exploration[np.isnan(exploration)] = 0
        exploration = np.power(exploration, self.alpha)

        q = agent.Q + exploration
        action = np.argmax(q)
        check = np.where(q == action)[0]
        if len(check) == 0:
            return action
        else:
            return np.random.choice(check)
