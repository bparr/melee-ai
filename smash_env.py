import numpy as np

# Number of inputs into the neural network.
SIZE_OF_STATE = 27


# (player number - 1) of our rl agent.
_RL_AGENT_INDEX = 1

_NUM_PLAYERS = 2


_NOTHING_ACTION = 0
_SHINE_ACTION = 1


# TODO these all assume no frame issues. Reconsider if training
#      does not compensate for the frame issues.
_SCRIPTS = (
    (0,),  # Nothing.
    (1, 1, 0, 2, 2, 0),  # Shine B. Jump out.
    ((3,) * 10 + (0,) * 12),  # Spot dodge.
    ((4,) * 10 + (0,) * 22),  # Roll left.
    ((5,) * 10 + (0,) * 22),  # Roll right.
)

_POST_SHINE_SCRIPTS = (
    (0,),  # Nothing.
    (1, 1, 0, 0, 0, 2, 2, 0),  # Multishine.
    ((3,) * 5 + (0,) * 8), # Wavedash down.
    ((4,) * 5 + (0,) * 8), # Wavedash left.
    ((5,) * 5 + (0,) * 8), # Wavedash right.
)

_MAX_EPISODE_LENGTH = 60 * 60


class SmashEnv():
    class _ActionSpace():
        def __init__(self):
            self.n = len(_SCRIPTS)

    def __init__(self):
        self.action_space = SmashEnv._ActionSpace()

        self._parser = None
        self._actionType = None

        self._frame_number = 0
        self._shine_last_action = False

        self._character = None  # This is set in make.
        self._pad = None  # This is set in make.
        self._opponent_character = None  # This is set in make.
        self._opponent_pad = None  # This is set in make.

    def get_game_length(self):
        return self._frame_number

    def make(self, args):
        return

    def step(self,action = None):
        action_to_script = _SCRIPTS
        if self._shine_last_action:
            action_to_script = _POST_SHINE_SCRIPTS
        self._shine_last_action = (action == _SHINE_ACTION)

        state, reward, is_terminal, env_done = None, 0.0, False, False
        for x in action_to_script[action]:
            if is_terminal or env_done:
                break
            state, intermediate_reward, is_terminal, env_done = self._step(x)
            reward += intermediate_reward

        # Only reward if intended to do nothing.
        if action != _NOTHING_ACTION or is_terminal:
            reward = 0.0

        return state, reward, is_terminal, env_done

    def _step(self, action=None):
        self._actionType.send(action, self._pad, self._character)

        match_state = None
        menu_state = None

        # Keep getting states until you reach a non-skipped frame
        while match_state is None and menu_state is None:
            match_state, menu_state = self.cpu.advance_frame()

        # Indicates that the match ended.
        if match_state is None:
            return [0.0] * SIZE_OF_STATE, 0.0, False, True

        self._frame_number += 1

        return self._parser.parse(match_state, self._frame_number,
                                  self._shine_last_action)

    def reset(self):
      raise Exception('reset')

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
