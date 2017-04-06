"""Core classes."""
import numpy as np
import random
import copy

WIDTH = 84
HEIGHT = 84


class ReplayMemory:
    """Store and replay (sample) memories."""
    def __init__(self, max_size, window_length):
        """Setup memory.

        You should specify the maximum size o the memory. Once the
        memory fills up oldest values are removed.
        """
        self._max_size = max_size
        self._window_length = window_length
        self._memory = []


    def append(self, old_state, reward, action, new_state, is_terminal):
        """Add a sample to the replay memory."""
        if len(self._memory) >= self._max_size:
            del(self._memory[0])
        self._memory.append((old_state, reward, action, new_state, is_terminal))


    def sample(self, batch_size, indexes=None):
        """Return samples from the memory.

        Returns
        --------
        (old_state_list, reward_list, action_list, new_state_list, is_terminal_list)
        """
        samples = random.sample(self._memory, min(batch_size, len(self._memory)))
        zipped = list(zip(*samples))
        zipped[0] = np.reshape(zipped[0], (-1, WIDTH, HEIGHT, self._window_length))
        zipped[3] = np.reshape(zipped[3], (-1, WIDTH, HEIGHT, self._window_length))
        return zipped


    def clear(self):
        """Reset the memory. Deletes all references to the samples."""
        self._memory = []

