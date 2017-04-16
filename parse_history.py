import numpy as np
from state import ActionState

# (player number - 1) of our rl agent.
_RL_AGENT_INDEX = 1

class Parser():

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

        is_terminal = players[_RL_AGENT_INDEX].percent > 0
        reward = 0

        # TODO Switch to rewarding ActionState.Wait (and other "waiting"
        #      action states??) so agent learns to not spam buttons.
        if not is_terminal:
            reward = 1

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
