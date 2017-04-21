from cpu import RESETTING_MATCH_STATE
from state import ActionState
import ssbm
import run
import numpy as np


# (player number - 1) of our rl agent.
_RL_AGENT_INDEX = 1

_NUM_PLAYERS = 2

_ACTION_TO_CONTROLLER_OUTPUT = [
    0,  # No button, neural control stick.
    27, # L + down (spot dodge, wave land, etc.)
]


class _Parser():
    def __init__(self):
        self.reset()

    def _get_action_state(self, state, player_index=_RL_AGENT_INDEX):
        return ActionState(state.players[player_index].action_state)

    def is_match_intro(self, state):
        action_state = self._get_action_state(state)
        return (action_state in [ActionState.Entry, ActionState.EntryStart,
                                 ActionState.EntryEnd])

    def parse(self, state):
        players = state.players[:_NUM_PLAYERS]

        # TODO Switch to rewarding ActionState.Wait (and other "waiting"
        #      action states??) so agent learns to not spam buttons.
        reward = 1.0

        is_terminal = players[_RL_AGENT_INDEX].percent > 0
        if is_terminal:
            reward = 0.0 #-256.0

        parsed_state = []

        for index in [0]:#range(_NUM_PLAYERS):
            player = players[index]
            parsed_state.append(float(player.action_state == 40))

            action_state = players[index].action_state
            if action_state != self._last_action_states[index]:
                self._last_action_states[index] = action_state
                self._frames_with_same_action[index] = -1
            self._frames_with_same_action[index] += 1
            # TODO change 3600.0 to something more reasonable? Or at least use MAX_EPISODE_LENGTH constant.
            parsed_state.append(float(self._frames_with_same_action[index]) / 3600.0)


        # Reshape so ready to be passed to network.
        parsed_state = np.reshape(parsed_state, (1, len(parsed_state)))

        return parsed_state, reward, is_terminal, None # debug_info

    def reset(self):
        self._last_action_states = [-1] * _NUM_PLAYERS
        self._frames_with_same_action = [0] * _NUM_PLAYERS


class SmashEnv():
    class _ActionSpace():
        def __init__(self):
            self.n = len(_ACTION_TO_CONTROLLER_OUTPUT)

    def __init__(self):
        self.action_space = SmashEnv._ActionSpace()

        self._parser = _Parser()

        # TODO Create a custom controller?
        self._actionType = ssbm.actionTypes['old']

        self._frame_number = 0
        self._character = None  # This is set in make.
        self._pad = None  # This is set in make.
        self._opponent_character = None  # This is set in make.
        self._opponent_pad = None  # This is set in make.

    def make(self, args):
        # Should only be called once
        self.cpu, self.dolphin = run.main(args)

        print("Running cpu.")
        self.cpu.run(dolphin_process=self.dolphin)
        self._character = self.cpu.characters[_RL_AGENT_INDEX]
        # Huh. cpu.py puts our pad at index 0 which != _RL_AGENT_INDEX.
        self._pad = self.cpu.pads[0]

        self._opponent_character = self.cpu.characters[0]
        self._opponent_pad = self.cpu.pads[1]

    def step(self,action = None):
        state, reward, is_terminal, debug_info = self._step(action)

        # Special case spot dodge to just wait until spotdodge is done.
        if not is_terminal and _ACTION_TO_CONTROLLER_OUTPUT[action] == 27:
            for i in range(21):
                asdf = 0 if i > 10 else 1
                # Use the No button action so can immediately spot dodge on next step.
                state, intermediate_reward, is_terminal, debug_info = self._step(0)
                reward += intermediate_reward
                if is_terminal:
                    reward = 0.0
                    break


        return state, reward, is_terminal, debug_info

    def _step(self, action=None):
        action = _ACTION_TO_CONTROLLER_OUTPUT[action]
        self._actionType.send(action, self._pad, self._character)

        opponent_action = 2
        if self._frame_number % 60 == 20:
            opponent_action = 7  # Down tilt
        self._actionType.send(opponent_action, self._opponent_pad, self._opponent_character)

        match_state = None
        menu_state = None

        # Keep getting states until you reach a non-skipped frame
        while match_state is None and menu_state is None:
            match_state, menu_state = self.cpu.advance_frame()

        # Indicates that the episode ended.
        # TODO shouldn't we return a state that is_terminal?!
        #      Only "fix" I can think of right now is an infinite time game.
        if match_state is None:
            match_state = self.reset()

        self._frame_number += 1

        return self._parser.parse(match_state)

    def reset(self):
        match_state = None
        menu_state = RESETTING_MATCH_STATE
        # Keep attempting to reset match until non-skipped non-reset frame.
        while ((match_state is None and menu_state is None) or
               menu_state == RESETTING_MATCH_STATE):
            match_state, menu_state = self.cpu.advance_frame(reset_match=True)

        # After episode has ended, just advance frames until the match starts.
        while (match_state is None or
               self._parser.is_match_intro(match_state)):
            match_state, menu_state = self.cpu.advance_frame()

        skipped_frames = 0
        while skipped_frames < 125:
            opponent_action = 2 if skipped_frames > 85 else 4  # Right (towards agent)
            self._actionType.send(opponent_action, self._opponent_pad, self._opponent_character)

            match_state, menu_state = self.cpu.advance_frame()
            if match_state is not None:
                skipped_frames += 1

        self._parser.reset()
        self._frame_number = 0
        return self._parser.parse(match_state)[0]

    def terminate(self):
        self.dolphin.terminate()


