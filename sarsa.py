import numpy as np
import math

def _coordinate_to_state(x, y):
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

    :type state: tuple
    :param state: State of the agent

    """
    state_x = np.clip(np.floor(x / 10), -7, 6) + 7
    state_y = np.clip(np.floor(y / 10), -1, 8) + 1
    # return int(state_x * 10 + state_y)
    return (state_x, state_y)

class FullModel(object):
    """
    Class including modules for parameter updates, choosing actions 
    and utility functions needed to implement a 
    simple RL model
    """
    def __init__(self, model = 'qlearning', reward_scheme = 'damage',num_states = 14*10, num_actions = 5, learning_rate = 0.1, 
                 exploration = 0.1, discount = 0.9, debug = True):
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
        self.reward_scheme = reward_scheme
        self._debug = debug

        self._all_states = [(x1, y1, x2, y2) for x1 in range(14) for y1 in range(10)
                            for x2 in range(14) for y2 in range(10)]

        # State value function
        self._q = {state: np.zeros(self._num_actions) for state in self._all_states}
        self._prev_state = (0, 0, 0, 0)

        # Damage recorded on previous frame
        self._prev_damage = 0
        
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

        return action

    def toggleExploration(self):
        """
        Switch exploration on/off
        """
        self._explore_on = not self._explore_on

        print("Explore state " + str(self._explore_on))

    def reward(self, x1, y1, x2, y2, damage):
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

        # if np.linalg.norm((x,y)) < 20:

        if self.reward_scheme == 'location':
            if np.linalg.norm((x1-x2,y1-y2)) < 20:
                # Near the center
                return -1
            elif x1 < -60 or x1 > 60:
                # Off-stage
                return -5
            else:
                return 0
        elif self.reward_scheme == 'damage':
            if self._prev_damage > 0 and damage == 0:
                return -1000
            else:
                return self._prev_damage - damage

    def update(self, cur_state, reward, cur_action):
        """
        Performs updates according to a specific algorithm

        Arguments:
        ----------

        :type cur_state: int
        :param cur_state: The current state of the agent

        :type reward: int
        :param reward: Reward received based on the previous state and action

        :type cur_action: int
        :param cur_action: The current action of the agent
        """
        best_action = np.argmax(self._q[cur_state])

        if self.model == 'qlearning':
            # Q-Learning : https://en.wikipedia.org/wiki/Q-learning
            self._q[self._prev_state][self._prev_action] += (self._learning_rate * (
                                                         reward + self._discount*self._q[cur_state][best_action]
                                                         - self._q[self._prev_state][self._prev_action]))

        elif self.model == 'sarsa':
            self._q[self._prev_state][self._prev_action] += (self._learning_rate * (
                                                         reward + self._discount*self._q[cur_state][cur_action]
                                                         - self._q[self._prev_state][self._prev_action]))

        elif self.model == 'expectedsarsa':
            expectation = 0
            for i in range(self._num_actions):
                if i == cur_action:
                    prob = 1 - self._exploration + self._exploration/self._num_actions
                else:
                    prob = self._exploration/self._num_actions

                expectation += prob * self._q[cur_state][i]

            self._q[self._prev_state][self._prev_action] += (self._learning_rate * (
                                                         reward + self._discount * expectation
                                                         - self._q[self._prev_state][self._prev_action]))

        self._prev_state = cur_state
        self._prev_action = cur_action
        self._cum_reward = 0.0001 * reward + 0.9999 * self._cum_reward

        if (self.frames_trained % 100 == 0):
            self.cum_reward_list.append(self._cum_reward)
        self.frames_trained += 1

        # Debugging and progress checking
        # print(self._q[0,3], self._q[70,0], self._q[130,4], self._cum_reward)

        if self._debug:
            if self.reward_scheme == 'location':
                print(self._q[(0,0)][3], self._q[(7,0)][0], self._q[(13,0)][4], self._cum_reward)
            elif self.reward_scheme == 'damage':
                print(self._q[(7,0,7,0)][0], self._q[(7,0,8,0)][3], self._q[(7,0,6,0)][4], self._cum_reward)

    def get_action(self, history):
        """
        Given exact x and y coordinates, returns the corresponding action 
        and performs appropriate RL updates 

        Arguments:
        ----------

        :type history: list
        :param history: List of recent states for all players

        Returns:
        --------

        :type action: int
        :param action: Action to take
        """

        x1 = history[-1].state.players[1].x
        y1 = history[-1].state.players[1].y
        x2 = history[-1].state.players[0].x
        y2 = history[-1].state.players[0].y
        damage = history[-1].state.players[1].percent
        lives = history[-1].state.players[1].stock

        cur_state = _coordinate_to_state(x1, y1) + _coordinate_to_state(x2, y2)

        reward = self.reward(x1, y1, x2, y2, damage)
        cur_action = self.act(cur_state)

        self.update(cur_state, reward, cur_action)
        self._prev_damage = damage

        return cur_action

    def calc_q(self, history):

        x1 = history[-1].state.players[1].x
        y1 = history[-1].state.players[1].y
        x2 = history[-1].state.players[0].x
        y2 = history[-1].state.players[0].y

        state = _coordinate_to_state(x1, y1) + _coordinate_to_state(x2, y2)

        return self._q[state]
