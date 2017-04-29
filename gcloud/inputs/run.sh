#!/usr/bin/env bash
#
# Depends on these environment variables being defined before executing:
#   - $MELEE_AI_ADDITIONAL_FLAGS
#   - $MELEE_AI_OUTPUT_PATH
#   - $MELEE_AI_GIT_REF

set -x  # echo commands to stdout
set -u  # throw an error if unset variable referenced
set -e  # exit on error

mkdir $MELEE_AI_OUTPUT_PATH

pushd melee-ai/
git checkout master  # TODO figure out if and why this is needed.
git pull --all
git checkout $MELEE_AI_GIT_REF

(time python3 dqn_atari.py --dolphin --iso ~/SSBM.iso --stage final_destination --is_worker --ai_output_dir=$MELEE_AI_OUTPUT_PATH $MELEE_AI_ADDITIONAL_FLAGS ) &> $MELEE_AI_OUTPUT_PATH/_worker_output.txt
popd

date > $MELEE_AI_OUTPUT_PATH/_done.txt

