import random
import numpy as np
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from gymnasium.spaces import Discrete

class QLearning:
    def __init__(self, num_actions: int, epsilon: float, alpha: float, gamma: float, initial_value=1.0):
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.initial_value = initial_value
        self.q_table = {}

    def sample_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        return self.argmax(state)

    def argmax(self, state):
        actions = [i for i in range(self.num_actions)]
        q_values = [self.get_q_value(state, action) for action in actions]
        max_q_value = max(q_values)
        max_actions = [action for action, value in enumerate(q_values) if value == max_q_value]
        return random.choice(max_actions)

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), self.initial_value)

    def learn(self, state, action, reward, next_state, done):
        current_q = self.get_q_value(state, action)
        if done:
            target = reward
        else:
            next_action = self.argmax(next_state)
            next_q = self.get_q_value(next_state, next_action)
            target = reward + self.gamma * next_q

        new_q = current_q + self.alpha * (target - current_q)
        self.q_table[(state, action)] = new_q
