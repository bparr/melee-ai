import numpy as np
from state import ActionState

# TODO make this a list of states??
_DOING_NOTHING_STATE = ActionState.Wait

# (player number - 1) of our rl agent.
_RL_AGENT_INDEX = 1

class Parser():

    def __init__(self):
        pass

    def _get_action_state(self, history, player_index=_RL_AGENT_INDEX):
        return ActionState(history[-1].state.players[player_index].action_state)

    def is_match_intro(self, history):
        action_state = self._get_action_state(history)
        return (action_state in [ActionState.Entry, ActionState.EntryStart,
                                 ActionState.EntryEnd])

    def parse(self, history):
        cur_state = history[-1].state.players[:2]

        is_terminal = cur_state[_RL_AGENT_INDEX].percent > 0
        reward = 0

        # TODO switch to doing nother reward once can show
        #      if agent learns to spame too much.
        # if self._get_action_state(history) == _DOING_NOTHING_STATE:
        if not is_terminal:
            reward = 1

        debug_info = history[-1].frame_counter

        state = []

        for index in range(len(cur_state)):
            for key,_ in cur_state[index]._fields:
                if key != 'controller':
                    val = getattr(cur_state[index], key)
                    if isinstance(val, bool):
                        val = int(val)
                    # print(key, val)
                    state.append(val)


        return np.array(state), reward, is_terminal, debug_info
