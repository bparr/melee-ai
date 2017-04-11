from smash_env import *
import pickle
import sarsa

env = SmashEnv()
history = env.make()
action = 0
prev_L = True

rl_model = sarsa.FullModel('qlearning', 'location')
# save_file = open("location_penalty_expectedsarsa_129000.pkl", 'rb')
# rl_model = pickle.load(save_file)

while True:
    if (history[-1].frame_counter % 4 == 1):
        action = rl_model.get_action(history)
        cur_L = history[-1].state.players[0].controller.button_L

        if (not prev_L) and cur_L:
            rl_model.toggleExploration()

        prev_L = cur_L
    
    history = env.step(action)

    if (history[-1].frame_counter % 36000 == 0):
        save_file = open(rl_model.reward_scheme+"_penalty_"+rl_model.model+"_"+str(rl_model.frames_trained)+".pkl", 'wb')
        pickle.dump(rl_model, save_file)
        save_file_2 = open(rl_model.reward_scheme+"_penalty_"+rl_model.model+"_cum_reward.pkl", 'wb')
        pickle.dump(rl_model.cum_reward_list, save_file_2)
        print("MODEL SAVED")