import time
from dolphin import DolphinRunner
from argparse import ArgumentParser
from multiprocessing import Process
from cpu import CPU
import RL
import util
import tempfile
import run
import numpy as np

class SmashEnv():

	def __init__(self):
		pass

	def make(self):
		# Should only be called once
		self.cpu, self.dolphin = run.main()
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

		return history

	def reset(self):
		history = 4
		while history == 4:
			history = self.cpu.advance_frame(reset_match=True)

		# After episode is ended just advance frames till match starts
		while (history == 2 or history == 3 or history == None):
			history = self.cpu.advance_frame(action=0)

		return history

	def terminate(self):
		self.dolphin.terminate()
