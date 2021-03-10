import random

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
        self.eps = 0.005
        self.gamma = 0.99
        self.alpha = 0.1

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if random.random() > self.eps:
            return np.argmax(self.Q[state])
        else:
            return np.random.choice(self.nA)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        current = self.Q[state][action]
        policy_s = np.ones(self.nA) * self.eps / self.nA        # current policy (for next state S')
        policy_s[np.argmax(self.Q[next_state])] += 1 - self.eps # greedy action
        Qsa_next = np.dot(self.Q[next_state], policy_s)         # get value of state at next time step
        target = reward + self.gamma * Qsa_next                 # construct target
        new_value = current + self.alpha * (target - current)   # get updated value 
    
        self.Q[state][action] = new_value