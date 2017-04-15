import numpy as np
import state

# TODO make this a list of states??
_DOING_NOTHING_STATE = state.ActionState.Wait

class Parser():

    def __init__(self):
        pass

    def parse(self, history):

        cur_state = history[-1].state.players[:2]

        is_terminal = cur_state[1].percent > 0
        reward = 0

        # TODO switch to doing nother reward once can show
        #      if agent learns to spame too much.
        #if cur_state[1].action_state == _DOING_NOTHING_STATE:
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
