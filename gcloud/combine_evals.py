import argparse
import numpy as np
import os
import pickle
import sys


WORKER_OUTPUT_EVALUATE_FILENAME = 'evaluate.p'

# Returns list of subdirectories of path.
# TODO this was copied from main.py. Reduce code redundancy.
def get_subdirs(parent_directory):
  input_dirs = os.listdir(parent_directory)
  input_dirs = [os.path.join(parent_directory, x) for x in input_dirs]
  input_dirs = sorted(x for x in input_dirs if os.path.isdir(x))
  return input_dirs


def main():
  script_directory = os.path.dirname(os.path.realpath(sys.argv[0]))
  parser = argparse.ArgumentParser(description='Combine output of WORKER MODE')
  parser.add_argument('-o', '--output-directory',
                      default=os.path.join(script_directory, 'outputs/'),
                      help='Location of evaluations.')
  args = parser.parse_args()


  models = get_subdirs(args.output_directory)
  for i, model in enumerate(models):
    model_evals = get_subdirs(model)
    all_rewards = []
    all_game_lengths = []
    for model_eval in model_evals:
      eval_filepath = os.path.join(
          model_eval, WORKER_OUTPUT_EVALUATE_FILENAME)
      with open(eval_filepath, 'rb') as f:
        rewards, game_lengths = pickle.load(f)

      all_rewards += rewards
      all_game_lengths += game_lengths

    print(i, np.mean(all_rewards), np.std(all_rewards), np.mean(all_game_lengths), np.std(all_game_lengths))



if __name__ == '__main__':
    main()
