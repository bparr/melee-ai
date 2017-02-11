import numpy as np
import math



class sarsa(object):

	def __init__(self, num_states = 14*10, num_actions = 5, learning_rate = 0.1, exploration = 0.1):
		self.num_states = num_states
		self.num_actions = num_actions
		self.exploration = exploration

		self.q = np.zeros((num_states,num_actions))
		self.prev_state = 0
		self.prev_action = 0
		self.learning_rate = learning_rate
		self.cum_reward = 0
		self.explore_on = True
		self.frames_trained = 0

	def act(self, state):

		if not self.explore_on or np.random.uniform(0,1) > self.exploration:
			action = np.argmax(self.q[state])
		else:
			action = np.random.choice(range(self.num_actions))

		return action

	def toggle(self):

		self.explore_on = not self.explore_on

		print("Explore state " + str(self.explore_on))

	def reward(self, x, y):

		if np.sqrt(pow(x,2) + pow(y,2)) < 20:
			reward = 1
		elif x < -60 or x > 60:
			reward = -10
		else:
			reward = 0

		return reward

	def update(self, cur_state, reward):

		best_action = np.argmax(self.q[cur_state])

		self.q[self.prev_state][self.prev_action] += (self.learning_rate * (
										             reward + self.q[cur_state][best_action]
										             - self.q[self.prev_state][self.prev_action]))

		self.prev_state = cur_state
		self.prev_action = best_action
		self.cum_reward = 0.01*reward + 0.99*self.cum_reward
		self.frames_trained += 1

		print(self.q[0,3], self.q[70,0], self.q[130,4], self.cum_reward)

	def coordinate_to_state(self, x, y):

		state_x = math.floor(x/10)

		if state_x < -6:
			state_x = -7
		elif state_x > 5:
			state_x = 6

		state_x = state_x + 7

		state_y = math.floor(y/10)

		if state_y < 0:
			state_y = -1

		if state_y > 7:
			state_y = 8

		state_y = state_y +1

		return state_x*10 + state_y