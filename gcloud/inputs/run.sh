#!/usr/bin/env bash

set -x  # echo commands to stdout
set -u  # throw an error if unset variable referenced
set -e  # exit on error

mkdir $MELEE_AI_SHARED_DIR/outputs

pushd melee-ai/
#TODO: git checkout correct thing.
# git checkout $MELEE_AI_GIT_REF

#TODO add --ai_input_dir=$MELEE_AI_SHARED_DIR/inputs/ --ai_output_dir=$MELEE_AI_SHARED_DIR/outputs/
python3 run.py --dolphin --load agents/FalconFalconBF/ --iso ~/SSBM.iso
popd

date > $MELEE_AI_SHARED_DIR/outputs/done.txt

