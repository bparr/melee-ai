import tensorflow as tf
import ssbm
import tf_lib as tfl
import numpy as np
from numpy import random, exp
import RL
import util
from default import *
from menu_manager import characters
import ctype_util as ct
import pprint

pp = pprint.PrettyPrinter(indent=2)

class Agent(Default):
  _options = [
    Option('char', type=str, choices=characters.keys(), help="character that this agent plays as"),
    Option('verbose', action="store_true", default=False, help="print stuff while running"),
    Option('reload', type=int, default=60, help="reload model every RELOAD seconds"),
    Option('dump', type=str, help="dump experiences to ip address via zmq"),
    Option('listen', type=str, help="address to listen on for model updates"),
  ]
  
  _members = [
    ('model', RL.Model)
  ]
  
  def __init__(self, **kwargs):
    kwargs = kwargs.copy()
    kwargs.update(mode=RL.Mode.PLAY)
    Default.__init__(self, **kwargs)
    
    self.frame_counter = 0
    self.action_counter = 0
    self.action = 0
    self.actions = util.CircularQueue(self.model.rlConfig.delay+1, 0)
    self.memory = util.CircularQueue(array=((self.model.memory+1) * ssbm.SimpleStateAction)())
    
    self.hidden = util.deepMap(np.zeros, self.model.model.hidden_size)
    
    self.model.restore()
    
    # TODO: merge dump and listen, they should always use the same address?
    if self.dump:
      try:
        import zmq
      except ImportError as err:
        print("ImportError: {0}".format(err))
        sys.exit("Install pyzmq to dump experiences")
      
      context = zmq.Context.instance()

      self.dump_socket = context.socket(zmq.PUSH)
      sock_addr = "tcp://%s:%d" % (self.dump, util.port(self.model.name + "/experience"))
      print("Connecting experience socket to " + sock_addr)
      self.dump_socket.connect(sock_addr)
      
      self.dump_size = self.model.rlConfig.experience_length
      self.dump_state_actions = (self.dump_size * ssbm.SimpleStateAction)()

      self.dump_frame = 0
      self.dump_count = 0

    if self.listen:
      import zmq
      context = zmq.Context.instance()
      self.params_socket = context.socket(zmq.SUB)
      self.params_socket.setsockopt(zmq.SUBSCRIBE, b"")
      address = "tcp://%s:%d" % (self.listen, util.port(self.model.name + "/params"))
      print("Connecting params socket to", address)
      self.params_socket.connect(address)

  def dump_state(self):
    state_action = self.dump_state_actions[self.dump_frame]
    state_action.state = self.state
    state_action.prev = self.prev_action
    state_action.action = self.action
    
    if self.dump_frame == 0:
      self.initial = self.hidden

    self.dump_frame += 1

    if self.dump_frame == self.dump_size:
      self.dump_count += 1
      self.dump_frame = 0
      
      if self.dump_count == 1:
        return # FIXME: figure out what is wrong with the first experience
      
      print("Dumping", self.dump_count)
      
      prepared = ssbm.prepareStateActions(self.dump_state_actions)
      prepared['initial'] = self.initial
      
      self.dump_socket.send_pyobj(prepared)

  def act(self, state, pad):
    self.frame_counter += 1
    if self.frame_counter % self.model.rlConfig.act_every != 0:
      return
    
    verbose = self.verbose and (self.action_counter % (10 * self.model.rlConfig.fps) == 0)
    #verbose = False
    
    self.state = state
    
    current = self.memory.peek()
    current.state = state
    
    self.prev_action = self.action
    current.prev_action = self.prev_action

    self.memory.increment()

    # History contains the entire state we need
    # history['state']['players'] is a list of 
    # 4 dictionaries, each being the complete state
    # of 1 player, with the state variables being
    # those in ssbm.PlayerMemory()

    history = self.memory.as_list()
    
    history = ct.vectorizeCTypes(ssbm.SimpleStateAction, history)
    history['hidden'] = self.hidden
    
    self.action, self.hidden = self.model.act(history, verbose)
    
    current.action = self.action

    #if verbose:
    #  pp.pprint(ct.toDict(state.players[1]))
    #  print(self.action)
    
    # the delayed action
    action = self.actions.push(self.action)
  
    action = 0
    # Jump three times then do nothing.
    # Use 400 instead 300 because first ten frames happen during
    # countdown, where can't control character.
    #if (self.frame_counter % 100) < 10 and self.frame_counter < 400:
    #  action = 20 # jump
    
    # # Constantly Up+B every 100 frames.
    # if (self.frame_counter % 100) < 10:
    #   # Huh that's super weird. Sometimes he jumps and then up b's instead of
    #   # just up b.
    #   action = 11  # up b

    ai_x = self.memory.as_list()[-1].state.players[1].x
    cpu_x = self.memory.as_list()[-1].state.players[0].x
    ai_facing = self.memory.as_list()[-1].state.players[1].facing
    cpu_facing = self.memory.as_list()[-1].state.players[0].facing

    # AI which tries to stay near the center of the stage
    # if (self.frame_counter % 20) < 10 and ai_x < -15:
    #   action = 4
    # elif (self.frame_counter % 20) < 10 and ai_x > 15:
    #   action = 3

    # AI which side+A attacks the opponent when they're nearby
    if (self.frame_counter % 30) < 20 and ai_x > cpu_x and ai_x-cpu_x < 20:
      action = 8
    elif (self.frame_counter % 30) < 20 and ai_x < cpu_x and cpu_x-ai_x < 20:
      action = 9      


    self.model.actionType.send(action, pad, self.char)
    
    self.action_counter += 1

    if self.dump:
      self.dump_state()
    
    if self.reload and self.action_counter % (self.reload * self.model.rlConfig.fps) == 0:
      if self.listen:
        import zmq
        blob = None
        
        # get the latest update from the trainer
        while True:
          try:
            #topic = self.socket.recv_string(zmq.NOBLOCK)
            blob = self.params_socket.recv_pyobj(zmq.NOBLOCK)
          except zmq.ZMQError as e:
            if e.errno == zmq.EAGAIN:
              # nothing to receive
              break
            # a real error
            raise e
        
        if blob is not None:
          print("unblobbing")
          self.model.unblob(blob)
        else:
          print("no blob received")
      else:
        self.model.restore()

