#!/bin/sh

python3 dqn_atari.py --gui --iso /home/bparr/dolphin/Super\ Smash\ Bros.\ Melee\ \(USA\)\ \(En\,Ja\)\ \(v1.02\).iso --cpu 9 --stage final_destination --is_worker --ai_input_dir ~/Desktop/dualing/inputs/1493471071.4083245/ --ai_output_dir=gcloud/user/ --worker_epsilon=0.0
