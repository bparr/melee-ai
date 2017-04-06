from copy import copy, deepcopy
import numpy as np

class Parser():

	def __init__(self):
		self.reset()

	def reset(self):
		self.prev_state = None

	def parse(self, history):

		cur_state = history[-1].state.players[:2]

		if self.prev_state is None:
			self.prev_state = deepcopy(history)


		reward = self.prev_state[-1].state.players[1].percent - cur_state[1].percent
		is_terminal = cur_state[1].stock < self.prev_state[-1].state.players[1].stock

		if is_terminal:
			reward = -1000

		debug_info = history[-1].frame_counter
		
		self.prev_state = deepcopy(history)

		state = []

		for index in range(len(cur_state)):
			for key,_ in cur_state[index]._fields:
				if key != 'controller':
					val = getattr(cur_state[index], key)
					if isinstance(val, bool):
						val = int(val)
					# print(key, val) 
					state.append(val)


		return np.array(state), reward, is_terminal, debug_info