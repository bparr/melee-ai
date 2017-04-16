from cpu import RESETTING_MATCH_STATE
from state import ActionState
import ssbm
import run
import numpy as np


# (player number - 1) of our rl agent.
_RL_AGENT_INDEX = 1

_ACTION_TO_CONTROLLER_OUTPUT = [
    0,  # No button, neural control stick.
    12, # Down B
    20, # Y (jump)
    25, # L (shield, air dodge)
    27, # L + down (spot dodge, wave land, etc.)
]



class _Parser():
    def __init__(self):
        pass

    def _get_action_state(self, state, player_index=_RL_AGENT_INDEX):
        return ActionState(state.players[player_index].action_state)

    def is_match_intro(self, state):
        action_state = self._get_action_state(state)
        return (action_state in [ActionState.Entry, ActionState.EntryStart,
                                 ActionState.EntryEnd])

    def parse(self, state):
        players = state.players[:2]

        # TODO Switch to rewarding ActionState.Wait (and other "waiting"
        #      action states??) so agent learns to not spam buttons.
        reward = 1

        is_terminal = players[_RL_AGENT_INDEX].percent > 0
        if is_terminal:
            reward = 0

        parsed_state = []
        for index in range(len(players)):
            for key,_ in players[index]._fields:
                if key != 'controller':
                    val = getattr(players[index], key)
                    if isinstance(val, bool):
                        val = int(val)
                    # print(key, val)
                    parsed_state.append(val)


        return np.array(parsed_state), reward, is_terminal, None # debug_info


class SmashEnv():
    class _ActionSpace():
        def __init__(self):
            self.n = len(_ACTION_TO_CONTROLLER_OUTPUT)

    def __init__(self):
        self.action_space = SmashEnv._ActionSpace()

        self._parser = _Parser()

        # TODO Create a custom controller?
        self._actionType = ssbm.actionTypes['old']

        self._character = None  # This is set in make.
        self._pad = None  # This is set in make.


    def make(self, args):
        # Should only be called once
        self.cpu, self.dolphin = run.main(args)

        print("Running cpu.")
        self.cpu.run(dolphin_process=self.dolphin)
        self._character = self.cpu.characters[_RL_AGENT_INDEX]
        # Huh. cpu.py puts our pad at index 0 which != _RL_AGENT_INDEX.
        self._pad = self.cpu.pads[0]

        return self.reset()

    def step(self,action = None):
        action = _ACTION_TO_CONTROLLER_OUTPUT[action]
        self._actionType.send(action, self._pad, self._character)

        match_state = None
        menu_state = None

        # Keep getting states until you reach a non-skipped frame
        while match_state is None and menu_state is None:
            match_state, menu_state = self.cpu.advance_frame()

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
            match_state, menu_state = self.cpu.advance_frame()

        return self._parser.parse(match_state)[0]

    def terminate(self):
        self.dolphin.terminate()
