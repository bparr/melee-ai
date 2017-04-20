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
    #12, # Down B
    #20, # Y (jump)
    # TODO reenable shield when needed.
    #25, # L (shield, air dodge)
    # TODO kirby has same spot dodge as fox.
    # TODO mewtwo has a great spot dodge, that would look awesome.
    # TODO if do change character, make sure to change special case for spotdodge in step!
    27, # L + down (spot dodge, wave land, etc.)
]

_KNOWN_ACTION_STATES = set([14.0, 15.0, 16.0, 17.0, 18.0, 20.0, 24.0, 25.0, 29.0, 35.0, 39.0, 41.0, 42.0, 43.0, 44.0, 45.0, 50.0, 53.0, 56.0, 57.0, 60.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 74.0, 178.0, 179.0, 180.0, 181.0, 182.0, 212.0, 213.0, 214.0, 215.0, 216.0, 217.0, 219.0, 220.0, 221.0, 222.0, 226.0, 227.0, 233.0, 235.0, 239.0, 240.0, 241.0, 242.0, 341.0, 343.0, 349.0, 351.0, 353.0, 356.0, 358.0, 360.0, 367.0, 368.0])
_ACTION_STATE_TO_INDEX = dict((x,i) for i, x in enumerate(sorted(_KNOWN_ACTION_STATES)))


# TODO add 'last action' history? Just prev action?
# Based on experiment listed at bottom of file.
_MEMORY_WHITELIST = [
    # TODO reenable if our agent can do damage.
    #'percent',
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
        reward = 1.0

        is_terminal = players[_RL_AGENT_INDEX].percent > 0
        if is_terminal:
            reward = -256.0

        parsed_state = []
        """
        print('')
        print('Our action state: %s, %s' % (players[1].action_state, ActionState(players[1].action_state)))
        print('Marth action state: %s, %s' % (players[0].action_state, ActionState(players[0].action_state)))
        print('Marth x: ' + str(players[0].x))
        """

        for index in range(_NUM_PLAYERS):
            # TODO just directly iterate over players?
            player = players[index]

            # Specific to Final Destination.
            #parsed_state.append((player.x + 250.0) / 500.0)
            #parsed_state.append((player.y + 150.0) / 300.0)
            # Based on Fox side B speed.
            #parsed_state.append((player.speed_air_x_self + 20.0) / 40.0)
            #parsed_state.append((player.speed_y_self + 20.0) / 40.0)
            # Mark unknown action state with index = len(_ACTION_STATE_TO_INDEX).
            action_state_index = _ACTION_STATE_TO_INDEX.get(player.action_state, len(_ACTION_STATE_TO_INDEX))
            parsed_state.append(1.0 * action_state_index / (1.0 + len(_ACTION_STATE_TO_INDEX)))

            #parsed_state.append(np.clip(player.facing, 0.0, 1.0))
            #parsed_state.append(float(player.charging_smash))
            #parsed_state.append(float(player.in_air))
            #parsed_state.append(player.shield_size / 60.0)
            # TODO what about kirby and jigglypuff?
            #parsed_state.append(player.jumps_used / 2.0)
            # TODO experiement for better normalizing constant. 60.0 was just a guess.
            #parsed_state.append(player.hitlag_frames_left / 60.0)

            """
            for key in _MEMORY_WHITELIST:
                val = getattr(players[index], key)
                parsed_state.append(float(val))
            """

            action_state = players[index].action_state
            if action_state != self._last_action_states[index]:
                self._last_action_states[index] = action_state
                self._frames_with_same_action[index] = 0
            self._frames_with_same_action[index] += 1
            # TODO change 3600.0 to something more reasonable? Or at least use MAX_EPISODE_LENGTH constant.
            parsed_state.append(float(self._frames_with_same_action[index]) / 3600.0)
            #parsed_state.append(float(ActionState(action_state) == ActionState.Escape))


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

        #self._opponent_last_state = None
        #self._opponent_last_state2 = None
        #self._dodge_count = 0

        self._spot_dodge_frame = 0


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
        if self._spot_dodge_frame > 0:
            self._spot_dodge_frame -= 1
            action = 0 if self._spot_dodge_frame < 10 else 1
        elif action == 1:
            # TODO Might be able to reduce by a frame or two. This appears consistent.
            self._spot_dodge_frame = 22

        state, reward, is_terminal, debug_info = self._step(action)

        """
        # Special case spot dodge to just wait until spotdodge is done.
        if not is_terminal and _ACTION_TO_CONTROLLER_OUTPUT[action] == 27:
            for _ in range(21):
                # Use the No button action so can immediately spot dodge on next step.
                state, intermediate_reward, is_terminal, debug_info = self._step(0)
                reward += intermediate_reward
                if is_terminal:
                    break
        """


        return state, reward, is_terminal, debug_info

    def _step(self, action=None):
        """
        action = 0
        if self._dodge_count > 0:
            self._dodge_count -=1
            action = 1
        elif self._opponent_last_state2 == ActionState.SquatWait and self._opponent_last_state == ActionState.AttackLw3:
            self._dodge_count = 10
            action = 1
        """

        action = _ACTION_TO_CONTROLLER_OUTPUT[action]
        self._actionType.send(action, self._pad, self._character)

        opponent_action = 2
        if self._frame_number % 60 == 20:
            #opponent_action = 5  # A only (jab)
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

        #self._opponent_last_state2 = self._opponent_last_state
        #self._opponent_last_state = ActionState(match_state.players[0].action_state)
        #print(self._opponent_last_state)
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
        # One frame shield: success (shield comes out).
        #     Eight frames of GuardReflect, then 15 frames of GuardOff.
        #
        # Frame 0 shield, frame 1 no shield. Frame 22 shield, frame 23 no shield --> Only a single shield.
        # Frame 0 shield, frame 1 no shield. Frame 23 shield, frame 24 no shield --> Two shields.

        # One frame spotdodge: success (spotdodge comes out)
        # action_state --> Escape for 22 frames (consistent with
        #     https://smashboards.com/threads/complete-fox-hitboxes-and-frame-data.285177/)
        # If done too early in match and still "Landing" instead of "Wait",
        # then need two frames, since a one frame will just shield.
        # In this two frame case, the first action state is GuardReflect
        # and the next frame is the correct Escape.

        # 50 is pretty large. 30 is safer. But cpu 9 Marth doesn't
        # dash over to fox, so cut a few more frames.
        #print(self._parser._get_action_state(match_state))
        while skipped_frames < 125:
            opponent_action = 0 if skipped_frames > 85 else 4  # Right (towards agent)
            self._actionType.send(opponent_action, self._opponent_pad, self._opponent_character)

            match_state, menu_state = self.cpu.advance_frame()
            if match_state is not None:
                skipped_frames += 1
                #print(self._parser._get_action_state(match_state))

                """
                if skipped_frames == 40:
                    action = _ACTION_TO_CONTROLLER_OUTPUT[2]
                    self._actionType.send(action, self._pad, self._character)
                if skipped_frames == 41:
                    action = _ACTION_TO_CONTROLLER_OUTPUT[0]
                    self._actionType.send(action, self._pad, self._character)
                """

        self._parser.reset()
        self._frame_number = 0
        #self._opponent_last_state = None
        #self._opponent_last_state2 = None
        #self._dodge_count = 0

        self._spot_dodge_frame = 0
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

