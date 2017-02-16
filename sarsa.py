import numpy as np
import math

class FullModel(object):
    """
    Class including modules for parameter updates, choosing actions 
    and utility functions needed to implement a 
    simple RL model
    """
    def __init__(self, model = 'qlearning', num_states = 14*10, num_actions = 5, learning_rate = 0.1, 
                 exploration = 0.1, discount = 0.9):
        """
        Arguments:
        ----------

        :type num_states: int
        :param num_states: Number of the states in RL setup

        :type num_actions: int
        :param num_actions: Number of the actions in RL setup

        :type learning_rate: float
        :param learning_rate: Learning rate to use in algorithm

        :type exploration: float
        :param exploration: Probability of exploration during each action selection instance

        """

        self._num_states = num_states
        self._num_actions = num_actions
        self._exploration = exploration
        self._learning_rate = learning_rate
        self.model = model
        self._discount = discount

        # State value function
        self._q = np.zeros((self._num_states, self._num_actions))
        self._prev_state = 0
        self._prev_action = 0

        # Cumulative regard to keep track of progress
        self._cum_reward = 0
        self.cum_reward_list=[]

        # Flag indicating whether or not to explore
        self._explore_on = True

        # Total number of updates so far
        self.frames_trained = 0

    def act(self, state):
        """
        Choose an action based on the value function (or exploration)
        and the current state

        Arguments:
        ----------

        :type state: int
        :param state: Current state

        Returns:
        --------

        :type action: int
        :param action: Action to take

        """
        if ((not self._explore_on) or (np.random.uniform(0,1) > self._exploration)):
            action = np.argmax(self._q[state])
        else:
            action = np.random.choice(range(self._num_actions))

        self._prev_action = action

        return action

    def toggleExploration(self):
        """
        Switch exploration on/off
        """
        self.explore_on = not self.explore_on

        print("Explore state " + str(self.explore_on))

    def reward(self, x, y):
        """
        Returns a reward based on the current state
        Arguments: 
        ----------

        :type x: float
        :param x: x-coordinate of agent

        :type y: float
        :param y: y-coordinate of agent

        Returns:
        --------

        :type reward: int
        :param reward: Reward based on the current state
        """

        if np.sqrt(pow(x,2) + pow(y,2)) < 20:
            # Near the center
            return 1
        elif x < -60 or x > 60:
            # Off-state
            return -1
        else:
            return 0

    def update(self, cur_state, reward):
        """
        Performs updates according to a specific algorithm

        Arguments:
        ----------

        :type cur_state: int
        :param cur_state: The current state of the agent

        :type reward: int
        :param reward: Reward received based on the previous state and action

        """
        best_action = np.argmax(self._q[cur_state])

        if self.model == 'qlearning':
            # Q-Learning : https://en.wikipedia.org/wiki/Q-learning
            self._q[self._prev_state][self._prev_action] += (self._learning_rate * (
                                                         reward + self._discount*self._q[cur_state][best_action]
                                                         - self._q[self._prev_state][self._prev_action]))

        self._prev_state = cur_state
        self._cum_reward = 0.0001 * reward + 0.9999 * self._cum_reward

        if (self.frames_trained % 100 == 0):
            self.cum_reward_list.append(self._cum_reward)
        self.frames_trained += 1

        # Debugging and progress checking
        print(self._q[0,3], self._q[70,0], self._q[130,4], self._cum_reward)

    def coordinate_to_state(self, x, y):
        """
        Maps exact x and y coordinates to a state that
        can be indexed in a list

        Arguments:
        ----------

        :type x: float
        :param x: x-coordinate of agent

        :type y: float
        :param y: y-coordinate of agent

        Returns:
        --------

        :type state: int
        :param state: State of the agent

        """
        state_x = np.clip(np.floor(x / 10), -7, 6) + 7
        state_y = np.clip(np.floor(y / 10), -1, 8) + 1
        return int(state_x * 10 + state_y)

    def get_action(self, x, y):
        """
        Given exact x and y coordinates, returns the corresponding action 
        and performs appropriate RL updates 

        Arguments:
        ----------

        :type x: float
        :param x: x-coordinate of agent

        :type y: float
        :param y: y-coordinate of agent

        Returns:
        --------

        :type action: int
        :param action: Action to take
        """

        cur_state = self.coordinate_to_state(x, y)
        reward = self.reward(x, y)
        self.update(cur_state, reward)
        return self.act(cur_state)
