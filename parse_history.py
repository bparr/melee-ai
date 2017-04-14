import numpy as np

# TODO make this a list of states??
_DOING_NOTHING_STATE = 14

class Parser():

    def __init__(self):
        pass

    def parse(self, history):

        cur_state = history[-1].state.players[:2]

        reward = 0
        if cur_state[1].action_state == _DOING_NOTHING_STATE:
            reward = 1
        is_terminal = cur_state[1].percent > 0

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
