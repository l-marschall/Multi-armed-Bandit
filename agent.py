import numpy as np
import pymc3 as pm


class Agent:
    """
    The Epsilon-Greedy policy will choose a random action with probability
    epsilon and take the best apparent approach with probability 1-epsilon. If
    multiple actions are tied for best choice, then a random action from that
    subset is selected.
    """

    def __init__(self, bandit, policy, prior=0):
        self.policy = policy
        self.N = bandit.N
        self.k = np.zeros(self.N, dtype=np.int)  # number of times action was chosen
        self.Q = prior * np.ones(self.N, dtype=np.float)  # estimated value
        self.last_action = None

    # Update Q action-value using:
    # Q(a) <- Q(a) + 1/(k+1) * (r(a) - Q(a))
    def update_Q(self, reward):
        self.k[self.last_action] += 1  # update action counter k -> k+1
        self.Q[self.last_action] += (1./self.k[self.last_action]) * \
            (reward - self.Q[self.last_action])

    def choose_action(self, bandit):
        action = self.policy.choose_action(bandit, self)
        self.last_action = action
        return action


class BetaAgent:
    """
    The Beta Agent is a Bayesian approach to a bandit problem with a Bernoulli
     or Binomial likelihood, as these distributions have a Beta distribution as
     a conjugate prior.
    """

    def __init__(self, bandit, policy):
        self.policy = policy
        self.N = bandit.N
        self.k = np.zeros(self.N, dtype=np.int)  # number of times action was chosen
        self.Q = np.zeros(self.N, dtype=np.float)  # estimated value
        self.last_action = None
        self.model = pm.Model()
        with self.model:
            self._prior = pm.Beta('prior', alpha=np.ones(self.N),
                                  beta=np.ones(self.N), shape=(1, self.N),
                                  transform=None)
        self._prior.distribution.alpha = np.ones(self.N)
        self._prior.distribution.beta = np.ones(self.N)

    # Update Q action-value using:
    # random draws from posterior distribution

    def update_Q(self, reward):
        self.k[self.last_action] += 1  # update action counter k -> k+1
        self.alpha[self.last_action] += reward
        self.beta[self.last_action] += 1 - reward
        self.Q = self._prior.random()

    def choose_action(self, bandit):
        action = self.policy.choose_action(bandit, self)
        self.last_action = action
        return action

    @property
    def alpha(self):
        return self._prior.distribution.alpha

    @property
    def beta(self):
        return self._prior.distribution.beta
