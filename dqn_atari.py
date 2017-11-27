#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import glob
#import gym
from smash_env import SmashEnv, SIZE_OF_STATE
import numpy as np
import os
import pickle
import random
import shutil
import sys
import tempfile
import tensorflow as tf
import time

from deeprl_hw2.core import ReplayMemory, mprint
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.objectives import mean_huber_loss
from deeprl_hw2.policy import GreedyPolicy, LinearDecayGreedyEpsilonPolicy, UniformRandomPolicy, GreedyEpsilonPolicy

RMSP_EPSILON = 0.01
RMSP_DECAY = 0.95
RMSP_MOMENTUM =0.95
EVAL_EPISODES = 5
CHECKPOINT_EVAL_EPISODES = 100
EVAL_CPU_LEVEL = 9

FIXED_SAMPLES_FILENAME = 'fixed_samples.p'
NUM_FIXED_SAMPLES = 1000


# TODO set to larger amount?
#MAX_EPISODE_LENGTH = 8 * 60 * 60 + 1000  # 1000 for just a little safety.
MAX_EPISODE_LENGTH =  999999999  # Basically disable this feature.
# Play between four to six minutes. Using random so workers don't continously
# start and stop at the same exact times.
PLAY_TOTAL_SECONDS = 5 * 60 + random.randint(-60, 60)
WORKER_EVALUATION_PROBABILITY = 0.02
WORKER_INPUT_MODEL_FILENAME = 'model.ckpt'
WORKER_INPUT_EPSILON_FILENAME = 'epsilon.txt'
WORKER_INPUT_RUN_SH_FILEPATH = 'gcloud/inputs/run.sh'
WORKER_OUTPUT_GAMEPLAY_FILENAME = 'memory.p'
WORKER_OUTPUT_EVALUATE_FILENAME = 'evaluate.p'

TOTAL_WORKER_JOBS = 10
NUM_BURN_IN_JOBS = 125 # TODO make sure this is reasonable.
# TODO experiment and ensure keeping up with workers' outputs.
FITS_PER_SINGLE_MEMORY = 1.0


# Number of gameplays between saving the model.
SAVE_MODEL_EVERY = 15


# Returns tuple of network, network_parameters.
def create_deep_q_network(input_frames, input_length, num_actions):
    input_frames_flat = tf.reshape(input_frames, [-1, input_length], name='input_frames_flat')
    fc1_W = tf.Variable(tf.random_normal([input_length, 128], stddev=0.1), name='fc1_W')
    fc1_b = tf.Variable(tf.zeros([128]), name='fc1_b')
    # (batch size, 256)
    output1 = tf.nn.relu(tf.matmul(input_frames_flat, fc1_W) + fc1_b, name='output1')

    fc2_W = tf.Variable(tf.random_normal([128, 256], stddev=0.1), name='fc2_W')
    fc2_b = tf.Variable(tf.zeros([256]), name='fc2_b')
    # (batch size, num_actions)
    output2 = tf.nn.relu(tf.matmul(output1, fc2_W) + fc2_b, name='output2')

    fc3_W = tf.Variable(tf.random_normal([256, num_actions], stddev=0.1), name='fc3_W')
    fc3_b = tf.Variable(tf.zeros([num_actions]), name='fc3_b')
    # (batch size, num_actions)
    q_network = tf.matmul(output2, fc3_W) + fc3_b

    network_parameters =  [fc1_W, fc1_b, fc2_W, fc2_b, fc3_W, fc3_b]

    return q_network, network_parameters


# Returns tuple of network, network_parameters.
def create_dual_q_network(input_frames, input_length, num_actions):
    input_frames_flat = tf.reshape(input_frames, [-1, input_length], name='input_frames_flat')
    W = tf.Variable(tf.random_normal([input_length, 128], stddev=0.1), name='W')
    b = tf.Variable(tf.zeros([128]), name='b')
    # (batch size, num_actions)
    output1 = tf.nn.relu(tf.matmul(input_frames_flat, W) + b, name='output1')

    fcV_W = tf.Variable(tf.random_normal([128, 512], stddev=0.1), name='fcV_W')
    fcV_b = tf.Variable(tf.zeros([512]), name='fcV_b')
    outputV = tf.nn.relu(tf.matmul(output1, fcV_W) + fcV_b, name='outputV')

    fcV2_W = tf.Variable(tf.random_normal([512, 1], stddev=0.1), name='fcV2_W')
    fcV2_b = tf.Variable(tf.zeros([1]), name='fcV2_b')
    outputV2 = tf.matmul(outputV, fcV2_W) + fcV2_b


    fcA_W = tf.Variable(tf.random_normal([128, 512], stddev=0.1), name='fcA_W')
    fcA_b = tf.Variable(tf.zeros([512]), name='fcA_b')
    outputA = tf.nn.relu(tf.matmul(output1, fcA_W) + fcA_b, name='outputA')

    fcA2_W = tf.Variable(tf.random_normal([512, num_actions], stddev=0.1), name='fcA2_W')
    fcA2_b = tf.Variable(tf.zeros([num_actions]), name='fcA2_b')
    outputA2 = tf.matmul(outputA, fcA2_W) + fcA2_b

    q_network = outputV2 + outputA2 - tf.reduce_mean(outputA2)

    network_parameters = [W, b, fcV_W, fcV_b, fcV2_W, fcV2_b, fcA_W, fcA_b, fcA2_W, fcA2_b]
    return q_network, network_parameters



def create_model(input_shape, num_actions, model_name, create_network_fn, learning_rate):  # noqa: D103
    """Create the Q-network model."""
    with tf.name_scope(model_name):
        input_frames = tf.placeholder(tf.float32, [None, input_shape],
                                      name ='input_frames')
        q_network, network_parameters = create_network_fn(
            input_frames, input_shape, num_actions)

        mean_max_Q =tf.reduce_mean( tf.reduce_max(q_network, axis=[1]), name='mean_max_Q')

        Q_vector_indexes = tf.placeholder(tf.int32, [None, 2], name ='Q_vector_indexes')
        gathered_outputs = tf.gather_nd(q_network, Q_vector_indexes, name='gathered_outputs')

        y_ph = tf.placeholder(tf.float32, name='y_ph')
        loss = mean_huber_loss(y_ph, gathered_outputs)
        train_step = tf.train.RMSPropOptimizer(learning_rate,
            decay=RMSP_DECAY, momentum=RMSP_MOMENTUM, epsilon=RMSP_EPSILON).minimize(loss)

    model = {
        'q_network' : q_network,
        'input_frames' : input_frames,
        'Q_vector_indexes' : Q_vector_indexes,
        'y_ph' : y_ph,
        'train_step': train_step,
        'mean_max_Q' : mean_max_Q,
    }
    return model, network_parameters



def calculate_mean_max_Q(sess, model, samples):
    mean_max = []
    INCREMENT = 1000
    for i in range(0, len(samples), INCREMENT):
        feed_dict = {model['input_frames']: samples[i: i + INCREMENT]}
        mean_max.append(sess.run(model['mean_max_Q'], feed_dict = feed_dict))
    return np.mean(mean_max)



def get_question_settings(question, batch_size):
    # if question == 2:
    #     return {
    #         'replay_memory_size': batch_size,
    #         'target_update_freq': None,
    #         'create_network_fn': create_linear_q_network,
    #         'is_double_network': False,
    #         'is_double_dqn': False,
    #     }

    # if question == 3:
    #     return {
    #         'replay_memory_size': 1000000,
    #         'target_update_freq': 10000,
    #         'create_network_fn': create_linear_q_network,
    #         'is_double_network': False,
    #         'is_double_dqn': False,
    #     }

    # if question == 4:
    #     return {
    #         'replay_memory_size': 1000000,
    #         'target_update_freq': None,
    #         'create_network_fn': create_linear_q_network,
    #         'is_double_network': True,
    #         'is_double_dqn': False,
    #     }

    if question == 5:
        return {
            'replay_memory_size': 1000000,
            'target_update_freq': 10000,
            'create_network_fn': create_deep_q_network,
            'is_double_network': False,
            'is_double_dqn': False,
        }

    if question == 6:
        return {
            'replay_memory_size': 1000000,
            'target_update_freq': 10000,
            'create_network_fn': create_deep_q_network,
            'is_double_network': False,
            'is_double_dqn': True,
        }

    if question == 7:
        return {
            'replay_memory_size': 1000000,
            'target_update_freq': 10000,
            'create_network_fn': create_dual_q_network,
            'is_double_network': False,
            'is_double_dqn': False,
        }



    raise Exception('Uknown question: ' + str(question))


def save_model(saver, sess, ai_input_dir, epsilon):
    # Use a temp dir so nothing tries to read half-written files.
    temp_dir = tempfile.mkdtemp(prefix='melee-ai-manager')
    saver.save(sess, os.path.join(temp_dir, WORKER_INPUT_MODEL_FILENAME))
    with open(os.path.join(temp_dir, WORKER_INPUT_EPSILON_FILENAME), 'w') as epsilon_file:
        epsilon_file.write(str(epsilon) + '\n')
    shutil.copy(WORKER_INPUT_RUN_SH_FILEPATH,
                os.path.join(temp_dir, os.path.basename(WORKER_INPUT_RUN_SH_FILEPATH)))

    shutil.move(temp_dir, os.path.join(ai_input_dir, str(time.time())))



def main():  # noqa: D103
    parser = argparse.ArgumentParser(description='Run DQN on Atari Space Invaders')
    parser.add_argument('--seed', default=10703, type=int, help='Random seed')
    parser.add_argument('--input_shape', default=SIZE_OF_STATE, help='Input shape')
    parser.add_argument('--gamma', default=0.99, help='Discount factor')
    # TODO experiment with this value.
    parser.add_argument('--epsilon', default=0.1, help='Final exploration probability in epsilon-greedy')
    parser.add_argument('--learning_rate', default=0.00025, help='Training learning rate.')
    parser.add_argument('--batch_size', default=32, type = int, help=
                                'Batch size of the training part')
    parser.add_argument('--question', type=int, default=7,
                        help='Which hw question to run.')


    parser.add_argument('--evaluate', action='store_true',
                        help='Only affects worker. Run evaluation instead of training.')
    parser.add_argument('--worker_epsilon', type=float,
                        help='Only affects worker. Override epsilon to use (instead of one in file).')
    parser.add_argument('--skip_model_restore', action='store_true',
                        help='Only affects worker. Use a newly initialized model instead of restoring one.')
    parser.add_argument('--generate_fixed_samples', action='store_true',
                        help=('Special case execution. Generate fixed samples and close. ' +
                             'This is necessary to run whenever the network or action space changes.'))
    parser.add_argument('--ai_input_dir', default='gcloud/inputs/',
                        help='Input directory with initialization files.')
    parser.add_argument('--ai_output_dir', default='gcloud/outputs/',
                        help='Output directory for gameplay files.')
    parser.add_argument('--is_worker', dest='is_manager',
                        action='store_false',
                        help='Whether this is a worker (no training).')
    parser.add_argument('--is_manager', dest='is_manager',
                        action='store_true',
                        help='Whether this is a manager (trains).')
    parser.set_defaults(is_manager=True)


    parser.add_argument('--psc', action='store_true',
                        help=('Only affects manager. Whether on PSC, ' +
                              'and should for example reduce disk usage.'))

    # Copied from original phillip code (run.py).
    #for opt in CPU.full_opts():
    #  opt.update_parser(parser)
    parser.add_argument("--dolphin", action="store_true", default=None, help="run dolphin")
    #for opt in DolphinRunner.full_opts():
    #  opt.update_parser(parser)

    args = parser.parse_args()
    # run.sh might pass these in via environment variable, so user directory
    # might not already be expanded.
    args.ai_input_dir = os.path.expanduser(args.ai_input_dir)
    args.ai_output_dir = os.path.expanduser(args.ai_output_dir)
    if args.is_manager:
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.set_random_seed(args.seed)

    do_evaluation = args.evaluate or random.random() < WORKER_EVALUATION_PROBABILITY
    if do_evaluation or args.generate_fixed_samples:
        args.cpu = EVAL_CPU_LEVEL
        print('OVERRIDING cpu level to: ' + str(EVAL_CPU_LEVEL))

    if args.generate_fixed_samples and args.is_manager:
        raise Exception('Can not generate fixed samples as manager. Must use ' +
                        '--is_worker and all other necessary flags (e.g. --iso ISO_PATH)')

    env = SmashEnv()
    if not args.is_manager:
        env.make(args)  # Opens Dolphin.

    question_settings = get_question_settings(args.question, args.batch_size)

    online_model, online_params = create_model(
        input_shape=args.input_shape,
        num_actions=env.action_space.n, model_name='online_model',
        create_network_fn=question_settings['create_network_fn'],
        learning_rate=args.learning_rate)

    target_model = online_model
    update_target_params_ops = []
    if (question_settings['target_update_freq'] is not None or
        question_settings['is_double_network']):
        target_model, target_params = create_model(
            input_shape=args.input_shape,
            num_actions=env.action_space.n, model_name='target_model',
            create_network_fn=question_settings['create_network_fn'],
            learning_rate=args.learning_rate)
        update_target_params_ops = [t.assign(s) for s, t in zip(online_params, target_params)]


    replay_memory = ReplayMemory(
        max_size=question_settings['replay_memory_size'],
        error_if_full=(not args.is_manager))


    saver = tf.train.Saver(max_to_keep=None)
    agent = DQNAgent(online_model=online_model,
                    target_model = target_model,
                    memory=replay_memory,
                    gamma=args.gamma,
                    target_update_freq=question_settings['target_update_freq'],
                    update_target_params_ops=update_target_params_ops,
                    batch_size=args.batch_size,
                    is_double_network=question_settings['is_double_network'],
                    is_double_dqn=question_settings['is_double_dqn'])

    sess = tf.Session()

    with sess.as_default():
        if args.generate_fixed_samples:
            print('Generating ' + str(NUM_FIXED_SAMPLES) + ' fixed samples and saving to ./' + FIXED_SAMPLES_FILENAME)
            print('This file is only ever used on the manager.')
            agent.compile(sess)
            fix_samples = agent.prepare_fixed_samples(
                env, sess, UniformRandomPolicy(env.action_space.n),
                NUM_FIXED_SAMPLES, MAX_EPISODE_LENGTH)
            env.terminate()
            with open(FIXED_SAMPLES_FILENAME, 'wb') as f:
                pickle.dump(fix_samples, f)
            return

        if args.is_manager or args.skip_model_restore:
            agent.compile(sess)
        else:
            saver.restore(sess, os.path.join(args.ai_input_dir, WORKER_INPUT_MODEL_FILENAME))

        print('_________________')
        print('number_actions: ' + str(env.action_space.n))

        # Worker code.
        if not args.is_manager:
          print('ai_input_dir: ' + args.ai_input_dir)
          print('ai_output_dir: ' + args.ai_output_dir)

          if do_evaluation:
              evaluation = agent.evaluate(env, sess, GreedyPolicy(), EVAL_EPISODES, MAX_EPISODE_LENGTH)
              print('Evaluation: ' + str(evaluation))
              with open(FIXED_SAMPLES_FILENAME, 'rb') as fixed_samples_f:
                fix_samples = pickle.load(fixed_samples_f)
              mean_max_Q = calculate_mean_max_Q(sess, online_model, fix_samples)

              evaluation = evaluation + (mean_max_Q,)
              with open(os.path.join(args.ai_output_dir, WORKER_OUTPUT_EVALUATE_FILENAME), 'wb') as f:
                  pickle.dump(evaluation, f)
              env.terminate()
              return

          worker_epsilon = args.worker_epsilon
          if worker_epsilon is None:
              with open(os.path.join(args.ai_input_dir, WORKER_INPUT_EPSILON_FILENAME)) as f:
                  lines = f.readlines()
                  # TODO handle unexpected lines better than just ignoring?
                  worker_epsilon = float(lines[0])
          print('Worker epsilon: ' + str(worker_epsilon))
          train_policy = GreedyEpsilonPolicy(worker_epsilon)

          agent.play(env, sess, train_policy, total_seconds=PLAY_TOTAL_SECONDS, max_episode_length=MAX_EPISODE_LENGTH)
          replay_memory.save_to_file(os.path.join(args.ai_output_dir, WORKER_OUTPUT_GAMEPLAY_FILENAME))
          env.terminate()
          return



        # Manager code.
        mprint('Loading fix samples')
        with open(FIXED_SAMPLES_FILENAME, 'rb') as fixed_samples_f:
            fix_samples = []#pickle.load(fixed_samples_f)
        print(args.ai_output_dir)

        evaluation_dirs = set()
        play_dirs = set()
        save_model(saver, sess, args.ai_input_dir, epsilon=1.0)
        epsilon_generator = LinearDecayGreedyEpsilonPolicy(
            1.0, args.epsilon, TOTAL_WORKER_JOBS / 5.0)
        fits_so_far = 0
        mprint('Begin to train (now safe to run gcloud)')
        mprint('Initial mean_max_q: ' + str(calculate_mean_max_Q(sess, online_model, fix_samples)))

        while len(play_dirs) < TOTAL_WORKER_JOBS:
            output_dirs = os.listdir(args.ai_output_dir)
            output_dirs = [os.path.join(args.ai_output_dir, x) for x in output_dirs]
            output_dirs = set(x for x in output_dirs if os.path.isdir(x))
            new_dirs = sorted(output_dirs - evaluation_dirs - play_dirs)

            if len(new_dirs) == 0:
                time.sleep(0.1)
                continue

            new_dir = new_dirs[-1]  # Most recent gameplay.
            evaluation_path = os.path.join(new_dir, WORKER_OUTPUT_EVALUATE_FILENAME)

            if os.path.isfile(evaluation_path):
                evaluation_dirs.add(new_dir)
                with open(evaluation_path, 'rb') as evaluation_file:
                    rewards, game_lengths, mean_max_Q = pickle.load(evaluation_file)
                evaluation = [np.mean(rewards), np.std(rewards),
                              np.mean(game_lengths), np.std(game_lengths),
                              mean_max_Q]
                mprint('Evaluation: ' + '\t'.join(str(x) for x in evaluation))
                continue

            memory_path = os.path.join(new_dir, WORKER_OUTPUT_GAMEPLAY_FILENAME)
            try:
                if os.path.getsize(memory_path) == 0:
                    # TODO Figure out why this happens despite temporary directory work.
                    #      Also sometimes the file doesn't exist? Hence the try/except.
                    mprint('Output not ready somehow: ' + memory_path)
                    time.sleep(0.1)
                    continue

                with open(memory_path, 'rb') as memory_file:
                    worker_memories = pickle.load(memory_file)
            except Exception as exception:
                print('Error reading ' + memory_path + ': ' + str(exception.args))
                time.sleep(0.1)
                continue
            for worker_memory in worker_memories:
                replay_memory.append(*worker_memory)
            if args.psc:
                os.remove(memory_path)


            play_dirs.add(time.time())
            #play_dirs.add(new_dir)
            #if len(play_dirs) <= NUM_BURN_IN_JOBS:
            #    mprint('Skip training because still burn in.')
            #    mprint('len(worker_memories): ' + str(len(worker_memories)))
            #    continue

            for _ in range(int(len(worker_memories) * FITS_PER_SINGLE_MEMORY)):
                agent.fit(sess, fits_so_far)
                fits_so_far += 1

            # Partial evaluation to give frequent insight into agent progress.
            # Last time checked, this took ~0.1 seconds to complete.
            mprint('mean_max_q, len(worker_memories): ' +
                   str(calculate_mean_max_Q(sess, online_model, fix_samples)) +
                   ', ' + str(len(worker_memories)))

            # Always decrement epsilon (e.g. not just when saving model).
            model_epsilon = epsilon_generator.get_epsilon(decay_epsilon=True)
            #if len(play_dirs) % SAVE_MODEL_EVERY == 0:
            #    save_model(saver, sess, args.ai_input_dir, model_epsilon)




if __name__ == '__main__':
    main()
