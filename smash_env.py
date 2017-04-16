import time
from dolphin import DolphinRunner
from argparse import ArgumentParser
from multiprocessing import Process
from cpu import RESETTING_MATCH_STATE
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

        self._parser = Parser()


    def make(self, args):
        # Should only be called once
        self.cpu, self.dolphin = run.main(args)

        print("Running cpu.")
        self.cpu.run(dolphin_process=self.dolphin)
        return self.reset()

    def step(self,action = None):
        match_state = None
        menu_state = None

        # Keep getting states until you reach a non-skipped frame
        while match_state is None and menu_state is None:
            match_state, menu_state = self.cpu.advance_frame(ACTION_TO_CONTROLLER_OUTPUT[action])

        # Indicates that the episode ended
        if match_state is None:
            match_state = self.reset()

        return self._parser.parse(match_state)

    def reset(self):
        match_state = None
        menu_state = 4
        # Keep attempting to reset match until non-skipped non-reset frame.
        while ((match_state is None and menu_state is None) or
               menu_state == RESETTING_MATCH_STATE):
            match_state, menu_state = self.cpu.advance_frame(reset_match=True)

        # After episode has ended, just advance frames until the match starts.
        while (match_state is None or
               self._parser.is_match_intro(match_state)):
            match_state, menu_state = self.cpu.advance_frame(action=0)

        return self._parser.parse(match_state)[0]

    def terminate(self):
        self.dolphin.terminate()
