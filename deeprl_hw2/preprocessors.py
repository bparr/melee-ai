"""Preprocessors for Atari pixel output."""

import numpy as np
from PIL import Image

from deeprl_hw2.core import HEIGHT, WIDTH
SIZE_OF_STATE = 9

class HistoryPreprocessor:
    """Keeps the last k states.

    Useful for domains where you need velocities, but the state
    contains only positions.

    When the environment starts, this will just fill the initial
    sequence values with zeros k times.

    Parameters
    ----------
    history_length: int
      Number of previous states to prepend to state being processed.

    """

    def __init__(self, history_length=1):
        self.history = np.zeros(shape=(1, , history_length+1))
        self.history_length = history_length


    def process_state_for_memory(self, state):
        """You only want history when you're deciding the current action to take.

        Returns last history_length processed states, where each is the max of
        the raw state and the previous raw state.
        """
        self.history[0,:,:,1:]=self.history[0,:,:,:self.history_length]
        self.history[0,:,:,0]=state

        result = np.zeros(shape=(1, WIDTH, HEIGHT, self.history_length), dtype = np.uint8)

        for i in range(self.history_length):
            result[0,:,:,i]=np.maximum(self.history[0,:,:,i], self.history[0,:,:,i+1])
        return result

    def reset(self):
        """Reset the history sequence.

        Useful when you start a new episode.
        """
        self.history = np.zeros(shape=(1, WIDTH, HEIGHT, self.history_length+1), dtype = np.uint8)


class PreprocessorSequence:
    """Combination of both an Atari preprocessor and history preprocessor."""
    def __init__(self, history_preprocessor):
        self._history_preprocessor = history_preprocessor

    def process_state_for_memory(self, state):
        state = self._history_preprocessor.process_state_for_memory(state)
        return state

    def reset(self):
        self._history_preprocessor.reset()

    def process_reward(self, reward):
        """Get sign of reward: -1, 0 or 1."""
        return reward

    def state2float(self, state):
        return state.astype(np.float32)


