from cpu import RESETTING_MATCH_STATE
from state import ActionState
from ssbm import SimpleAction, SimpleButton, SimpleController
import run
import numpy as np
import pyautogui
import time

# Number of inputs into the neural network.
SIZE_OF_STATE = 27


# (player number - 1) of our rl agent.
_RL_AGENT_INDEX = 1

_NUM_PLAYERS = 2

_CONTROLLER = [
    SimpleController(SimpleButton.NONE, (0.5, 0.5)), # Neutral.
    SimpleController(SimpleButton.B, (0.5, 0.0)), # Shine.
    SimpleController(SimpleButton.Y, (0.5, 0.0)), # Jump.
    SimpleController(SimpleButton.L, (0.5, 0.0)), # Dodge/wavedash down.
    # Use slightly below 0.5 for controller Y-axis, so can wavedash
    # far. Note that using 0.5 exactly does not allow wavedashing.
    # Based on https://www.reddit.com/r/smashbros/comments/2bbw7m/longest_and_shortest_wavedash/
    SimpleController(SimpleButton.L, (0.0, 89.0 / 255.0)), # Dodge/wavedash left.
    SimpleController(SimpleButton.L, (1.0, 89.0 / 255.0)), # Dodge/wavedash right.
]


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

_MAX_EPISODE_LENGTH = 99 * 60 * 60 # Disabled for demo.

class _Parser():
    def __init__(self):
        self.reset()

    def _get_action_state(self, state, player_index=_RL_AGENT_INDEX):
        return ActionState(state.players[player_index].action_state)

    def is_match_intro(self, state):
        action_state = self._get_action_state(state)
        return (action_state in [ActionState.Entry, ActionState.EntryStart,
                                 ActionState.EntryEnd])

    def parse(self, state, frame_number, shine_last_action):
        players = state.players[:_NUM_PLAYERS]

        reward = 0.0
        if ActionState(players[_RL_AGENT_INDEX].action_state) == ActionState.Wait:
            reward = 1.0 / 60.0

        # TODO add this case? abs(players[_RL_AGENT_INDEX].x) >= 87.5)
        is_terminal = (players[_RL_AGENT_INDEX].percent > 0 or
                       players[_RL_AGENT_INDEX].stock != 4)
        if is_terminal:
            reward = 0.0 #-256.0

        env_done = is_terminal or (frame_number >= _MAX_EPISODE_LENGTH)

        parsed_state = []

        for index in range(_NUM_PLAYERS):
            player = players[index]

            # Specific to Final Destination.
            parsed_state.append((player.x + 250.0) / 500.0)
            parsed_state.append((player.y + 150.0) / 300.0)

            # Based on Fox side B speed.
            parsed_state.append((player.speed_air_x_self + 20.0) / 40.0)
            parsed_state.append((player.speed_y_self + 20.0) / 40.0)

            parsed_state.append(float(player.action_state) / 382.0)

            parsed_state.append(np.clip(player.facing, 0.0, 1.0))
            parsed_state.append(float(player.charging_smash))
            parsed_state.append(float(player.in_air))
            parsed_state.append(player.shield_size / 60.0)
            # TODO what about kirby and jigglypuff?
            parsed_state.append(player.jumps_used / 2.0)
            # TODO experiement for better normalizing constant. 60.0 was just a guess.
            parsed_state.append(player.hitlag_frames_left / 60.0)
            parsed_state.append(player.percent / 1000.0)


            action_state = players[index].action_state
            if action_state != self._last_action_states[index]:
                self._last_action_states[index] = action_state
                self._frames_with_same_action[index] = -1
            self._frames_with_same_action[index] += 1
            # TODO change _MAX_EPISODE_LENGTH to something more reasonable?
            parsed_state.append(float(self._frames_with_same_action[index]) / (1.0 * _MAX_EPISODE_LENGTH))


        parsed_state.append(float(shine_last_action))

        # Reshape so ready to be passed to network.
        parsed_state = np.reshape(parsed_state, (1, len(parsed_state)))

        return parsed_state, reward, is_terminal, env_done

    def reset(self):
        self._last_action_states = [-1] * _NUM_PLAYERS
        self._frames_with_same_action = [0] * _NUM_PLAYERS


class SmashEnv():
    class _ActionSpace():
        def __init__(self):
            self.n = len(_SCRIPTS)

    def __init__(self):
        self.action_space = SmashEnv._ActionSpace()

        self._parser = _Parser()
        self._actionType = SimpleAction(_CONTROLLER)

        self._frame_number = 0
        self._shine_last_action = False

        self._character = None  # This is set in make.
        self._pad = None  # This is set in make.
        self._opponent_character = None  # This is set in make.
        self._opponent_pad = None  # This is set in make.

    def get_game_length(self):
        return self._frame_number

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
        pyautogui.keyDown('f1')
        match_state = None
        menu_state = None

        # After episode has ended, just advance frames until the match resets.
        while (match_state is None or match_state.frame > 150):
            match_state, menu_state = self.cpu.advance_frame()

        pyautogui.keyUp('f1')
        self._parser.reset()
        self._frame_number = 0
        self._shine_last_action = False

        return self._parser.parse(match_state, self._frame_number,
                                  self._shine_last_action)[0]

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
