import numpy as np
import math

class sarsa(object):
	"""
	Class including modules for parameter updates, choosing actions 
	and utility functions needed to implement a 
	simple RL model
	"""
	def __init__(self, num_states = 14*10, num_actions = 5, learning_rate = 0.1, exploration = 0.05):
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

		self.num_states = num_states
		self.num_actions = num_actions
		self.exploration = exploration
		self.learning_rate = learning_rate

		# State value function
		self.q = np.zeros((num_states,num_actions))
		self.prev_state = 0
		self.prev_action = 0

		# Cumulative regard to keep track of progress
		self.cum_reward = 0

		# Flag indicating whether or not to explore
		self.explore_on = True

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

		if ((not self.explore_on) or np.random.uniform(0,1)) > self.exploration:
			action = np.argmax(self.q[state])
		else:
			action = np.random.choice(range(self.num_actions))

		self.prev_action = action

		return action

	def toggle(self):
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
			reward = 1
		elif x < -60 or x > 60:
			# Off-state
			reward = -1
		else:
			reward = 0

		return reward

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
		best_action = np.argmax(self.q[cur_state])

		# Q-Learning : https://en.wikipedia.org/wiki/Q-learning
		self.q[self.prev_state][self.prev_action] += (self.learning_rate * (
										             reward + self.q[cur_state][best_action]
										             - self.q[self.prev_state][self.prev_action]))

		self.prev_state = cur_state
		self.cum_reward = 0.0001*reward + 0.9999*self.cum_reward
		self.frames_trained += 1

		# Debugging and progress checking
		print(self.q[0,3], self.q[70,0], self.q[130,4], self.cum_reward)

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
		state_x = math.floor(x/10)

		if state_x < -7:
			state_x = -7
		elif state_x > 6:
			state_x = 6

		state_x = state_x + 7

		state_y = math.floor(y/10)

		if state_y < -1:
			state_y = -1

		if state_y > 8:
			state_y = 8

		state_y = state_y +1

		return state_x*10 + state_y