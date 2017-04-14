import time
from dolphin import DolphinRunner
from argparse import ArgumentParser
from multiprocessing import Process
from cpu import CPU
import util
import tempfile
import run
import numpy as np
from parse_history import Parser

NUM_OF_ACTION = 5

class SmashEnv():
	class _ActionSpace():
		def __init__(self):
			self.n = NUM_OF_ACTION

	def __init__(self):
		self.action_space = SmashEnv._ActionSpace()
		self.Parser = Parser()


	def make(self, parser):
    # Should only be called once
		self.cpu, self.dolphin = run.main(parser)

		print("Running cpu.")
		self.cpu.run(dolphin_process=self.dolphin)
		return self.reset()

	def step(self,action = None):
		history = None

		# Keep getting history until you reach a non-skipped frame
		while history is None:
			history = self.cpu.advance_frame(action)

		# Indicates that the episode ended

		# TODO rename numbers
		if history == 2 or history == 3 :
			history = self.reset()

		state, reward, is_terminal, debug_info = self.Parser.parse(history)
		return state, reward, is_terminal, debug_info

	def reset(self):
		history = 4
		while history == 4:
			history = self.cpu.advance_frame(reset_match=True)
    
		self.Parser.reset()


		# After episode is ended just advance frames till match starts
		while (history == 2 or history == 3 or history == None):
			history = self.cpu.advance_frame(action=0)

		state, reward, is_terminal, debug_info = self.Parser.parse(history)
		return state

	def terminate(self):
		self.dolphin.terminate()
