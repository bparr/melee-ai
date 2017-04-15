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

ACTION_TO_CONTROLLER_OUTPUT = [
    0,  # No button, neural control stick.
    12, # Down B
    20, # Y (jump)
    25, # L (shield, air dodge)
    27, # L + down (spot dodge, wave land, etc.)
]
NUM_OF_ACTION = len(ACTION_TO_CONTROLLER_OUTPUT)

class SmashEnv():
    class _ActionSpace():
        def __init__(self):
            self.n = NUM_OF_ACTION

    def __init__(self):
        self.action_space = SmashEnv._ActionSpace()
        self.parser = Parser()


    def make(self, args):
        # Should only be called once
        self.cpu, self.dolphin = run.main(args)

        print("Running cpu.")
        self.cpu.run(dolphin_process=self.dolphin)
        return self.reset()

    def step(self,action = None):
        history = None

        # Keep getting history until you reach a non-skipped frame
        while history is None:
            history = self.cpu.advance_frame(ACTION_TO_CONTROLLER_OUTPUT[action])

        # Indicates that the episode ended

        # TODO rename numbers
        if history == 2 or history == 3 :
            history = self.reset()

        state, reward, is_terminal, debug_info = self.parser.parse(history)
        return state, reward, is_terminal, debug_info

    def reset(self):
        history = 4
        while history == 4:
            history = self.cpu.advance_frame(reset_match=True)

        # After episode is ended just advance frames till match starts
        while (history == 2 or history == 3 or history == None or
               self.parser.is_match_intro(history)):
            history = self.cpu.advance_frame(action=0)

        state, reward, is_terminal, debug_info = self.parser.parse(history)
        return state

    def terminate(self):
        self.dolphin.terminate()
