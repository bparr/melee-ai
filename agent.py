import ssbm
import numpy as np
from numpy import random, exp
import ssbm
import util
from default import *
from menu_manager import characters
import pprint
import pickle

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


  def act(self, state, pad, action):
    # TODO seems we broke self.char. It is always None. Fix.
    self.actionType.send(action, pad, self.char)
