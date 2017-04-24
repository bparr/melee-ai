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
    # TODO reenable shine inputs when needed.
    12, # Down B
    20, # Y (jump)
    25, # L (shield, air dodge)
    27, # L + down (spot dodge, wave land, etc.)
]


# Based on experiment listed at bottom of file.
_MEMORY_WHITELIST = [
    'percent',
    'facing',  # 1.0 is right, -1.0 is left.
    'x',
    'y',
    'action_state',
    'hitlag_frames_left',
    'jumps_used',
    'speed_air_x_self',  # Also ground speed when not in_air.
    'speed_y_self',
    'shield_size',
    'charging_smash',
    'in_air',
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

        is_terminal = (players[_RL_AGENT_INDEX].percent > self._previous_damage) or (players[_RL_AGENT_INDEX].stock < self._previous_lives)
        match_over = players[_RL_AGENT_INDEX].stock == 0

        if self._previous_lives > players[_RL_AGENT_INDEX].stock:
            reward = -1000
        else:
            reward = self._previous_damage - players[_RL_AGENT_INDEX].percent

        self._previous_damage = players[_RL_AGENT_INDEX].percent
        self._previous_lives = players[_RL_AGENT_INDEX].stock

        # if is_terminal:
        #     reward = 0

        parsed_state = []
        for index in range(_NUM_PLAYERS):
            for key in _MEMORY_WHITELIST:
                val = getattr(players[index], key)
                parsed_state.append(float(val))

            action_state = players[index].action_state
            if action_state != self._last_action_states[index]:
                self._last_action_states[index] = action_state
                self._frames_with_same_action[index] = 0
            self._frames_with_same_action[index] += 1
            parsed_state.append(float(self._frames_with_same_action[index]))


        # Reshape so ready to be passed to network.
        parsed_state = np.reshape(parsed_state, (1, len(parsed_state)))

        return parsed_state, reward, is_terminal, match_over # debug_info

    def reset(self):
        self._last_action_states = [-1] * _NUM_PLAYERS
        self._frames_with_same_action = [0] * _NUM_PLAYERS
        self._previous_damage = 0
        self._previous_lives = 3


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

    def step(self,action = None):
        action = _ACTION_TO_CONTROLLER_OUTPUT[action]
        self._actionType.send(action, self._pad, self._character)

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

        self._parser.reset()
        return self._parser.parse(match_state)[0]

    def terminate(self):
        self.dolphin.terminate()


# I set SIDevice0 = 12 in Dolphin.ini and experiemented with each line
# individually. The results are below. Some notes:
#   - A charged smash attack has no extra action_state value. Just longer.
#     Combined with charging_smash staying True during hit of said smash
#     means the agent can never be sure when the attack is.
#     TODO improve!
#
#  - None of these say how long until next input is allowed. Twas a hope,
#    though in retrospect I'm not sure how Direction Influence would work if this
#    value was somehow available.

#Looks really good.
#print(players[0].percent)
#print(players[0].facing)
#print(players[0].x)
#print(players[0].y)  # Does not detect crouch. action_state does.
#print(players[0].action_state)
#print(players[0].hitlag_frames_left)
#print(players[0].jumps_used)  # Does not include up+b.
#print(players[0].speed_air_x_self)  # Not just while in air.
#print(players[0].speed_y_self)
# Starts at max of 60. Shield breaks at 0. After breaking it
# slowly goes up about halfway while player is stunned. When
# used, decreases faster than recovers when not used.
#print(players[0].shield_size)


# Looks good but remains True through attack, so no way to
# detect when hitbox will happen... :(
#print(players[0].charging_smash)

# Looks good but most likely useless to us.
#print(players[0].stock)
#print(players[0].character)
#print(players[0].speed_ground_x_self)  # Is 0 in air.
#print(players[0].in_air)  # Minimally useful on Final Destination.


# After actually being hit, then looks good, but sometimes it just goes bad.
# E.g. after a spot-dodge it stays at some int > 0.
#print(players[0].hitstun_frames_left)

# Seems to increment every time action_state changes. Starts at 0.
#print(players[0].action_counter)

# Counts number of frames in action mod 90(!).
#print(players[0].action_frame)

# Mainly false. Weirdly goes to True when turn into Sheik.
#print(players[0].invulnerable)


# Always seems to be constant.
#print(players[0].z)  # Always 0.
#print(players[0].speed_x_attack)  # Always 0.
#print(players[0].speed_y_attack)  # Always 0.
#print(players[0].cursor_x)  # Always -31.0.
#print(players[0].cursor_y)  # Always -21.5.

