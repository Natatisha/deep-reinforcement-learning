import numpy as np
from collections import defaultdict


class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.i_episode = 1.0
        self.epsilon = 1.0

    def update_Q(self, Qsa, Qsa_next, reward, alpha, gamma):
        return Qsa + (alpha * (reward + (gamma * Qsa_next) - Qsa))

    def get_policy(self, actions, epsilon=0.001):
        default_prob = epsilon / self.nA
        greedy_prob = (1.0 - epsilon) + default_prob
        best_action = np.argmax(actions)
        policy = np.ones(self.nA) * default_prob
        policy[best_action] = greedy_prob
        return policy

    def select_action(self, state, epsilon=0.01, use_decreasing_epsilon=True):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """

        self.epsilon = 1.0 / self.i_episode
        epsilon = epsilon if not use_decreasing_epsilon else self.epsilon
        return np.random.choice(np.arange(self.nA), p=self.get_policy(self.Q[state], epsilon))

    def step(self, state, action, reward, next_state, done, alpha=0.01, gamma=1):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        # expected sarsa
        policy_next = self.get_policy(self.Q[next_state], 0.1)
        # Q_next = 0 if done else sum([q * prob_a for q, prob_a in zip(self.Q[next_state], policy_next)])

        # q-learning
        Q_next = 0 if done else max(self.Q[next_state])

        self.Q[state][action] = self.update_Q(self.Q[state][action], Q_next, reward, alpha, gamma)

        if done:
            self.i_episode += 1
