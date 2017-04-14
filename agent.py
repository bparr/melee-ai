import ssbm
import numpy as np
from numpy import random, exp
import ssbm
import util
from default import *
from menu_manager import characters
import pprint
import pickle
import sarsa

pp = pprint.PrettyPrinter(indent=2)

class Agent(Default):
  _options = [
    Option('char', type=str, choices=characters.keys(), help="character that this agent plays as"),
    Option('verbose', action="store_true", default=False, help="print stuff while running"),
    Option('action_type', type=str, default="old", choices=ssbm.actionTypes.keys()),
  ]


  def __init__(self, **kwargs):
    kwargs = kwargs.copy()
    Default.__init__(self, **kwargs)

    self.actionType = ssbm.actionTypes[self.action_type]
    self.frame_counter = 0
    self.prev_action = 0
    self.memory = util.CircularQueue(array=((1) * ssbm.SimpleStateAction)())

    self.prev_L = False


  def act(self, state, pad, action):
    self.frame_counter += 1

    current = self.memory.peek()
    current.state = state
    current.prev_action = self.prev_action
    current.action = action

    self.prev_action = action

    self.memory.increment()

    # History contains the entire state we need
    # history['state']['players'] is a list of
    # 4 dictionaries, each being the complete state
    # of 1 player, with the state variables being
    # those in ssbm.PlayerMemory()

    self.actionType.send(action, pad, self.char)

    history = self.memory.as_list()
    history[-1].frame_counter = self.frame_counter

    return history
