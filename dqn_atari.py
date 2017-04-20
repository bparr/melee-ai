#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import glob
#import gym
from smash_env import SmashEnv
import numpy as np
import os
import pickle
import random
import shutil
import sys
import tempfile
import tensorflow as tf
import time

from dolphin import DolphinRunner
from cpu import CPU

from deeprl_hw2.core import ReplayMemory
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.objectives import mean_huber_loss
from deeprl_hw2.policy import GreedyPolicy, LinearDecayGreedyEpsilonPolicy, UniformRandomPolicy, GreedyEpsilonPolicy
from deeprl_hw2.core import SIZE_OF_STATE

RMSP_EPSILON = 0.01
RMSP_DECAY = 0.95
RMSP_MOMENTUM =0.95
EVAL_EPISODES = 10
CHECKPOINT_EVAL_EPISODES = 100

FIXED_SAMPLES_FILENAME = 'fixed_samples.p'
NUM_FIXED_SAMPLES = 1000


# TODO set to larger amount?
#MAX_EPISODE_LENGTH = 8 * 60 * 60 + 1000  # 1000 for just a little safety.
MAX_EPISODE_LENGTH =  60 * 60  # One minute.
NUM_WORKER_FRAMES = MAX_EPISODE_LENGTH
WORKER_EVALUATION_PROBABILITY = 0.02
WORKER_INPUT_MODEL_FILENAME = 'model.ckpt'
WORKER_INPUT_EPSILON_FILENAME = 'epsilon.txt'
WORKER_INPUT_RUN_SH_FILEPATH = 'gcloud/inputs/run.sh'
WORKER_OUTPUT_GAMEPLAY_FILENAME = 'memory.p'
WORKER_OUTPUT_EVALUATE_FILENAME = 'evaluate.p'

TOTAL_WORKER_JOBS = 2000
NUM_BURN_IN_JOBS = int(50000 / MAX_EPISODE_LENGTH)
# TODO experiment and ensure keeping up with workers' outputs.
FIT_PER_JOB = 1000



# Returns tuple of network, network_parameters.
def create_linear_q_network(input_frames, input_length, num_actions):
    input_frames_flat = tf.reshape(input_frames, [-1, input_length], name='input_frames_flat')
    W = tf.Variable(tf.random_normal([input_length, num_actions], stddev=0.025))
    b = tf.Variable(tf.zeros([num_actions]))
    # (batch size, num_actions)
    q_network = tf.matmul(input_frames_flat, W) + b

    return q_network, [W, b]


# Returns tuple of flat output, flat output size, network_parameters.
def create_conv_network(input_frames):
    conv1_W = tf.Variable(tf.random_normal([8, 8, 4, 16], stddev=0.1), name='conv1_W')
    conv1_b = tf.Variable(tf.zeros([16]), name='conv1_b')
    conv1 = tf.nn.conv2d(input_frames, conv1_W, strides=[1, 4, 4, 1], padding='VALID', name='conv1')
    # (batch size, 20, 20, 16)
    output1 = tf.nn.relu(conv1 + conv1_b, name='output1')

    conv2_W = tf.Variable(tf.random_normal([4, 4, 16, 32], stddev=0.1), name='conv2_W')
    conv2_b = tf.Variable(tf.zeros([32]), name='conv2_b')
    conv2 = tf.nn.conv2d(output1, conv2_W, strides=[1, 2, 2, 1], padding='VALID', name='conv2')
    # (batch size, 9, 9, 32)
    output2 = tf.nn.relu(conv2 + conv2_b, name='output2')

    flat_output2_size = 9 * 9 * 32
    flat_output2 = tf.reshape(output2, [-1, flat_output2_size], name='flat_output2')

    return flat_output2, flat_output2_size, [conv1_W, conv1_b, conv2_W, conv2_b]


# Returns tuple of network, network_parameters.
def create_deep_q_network(input_frames, input_length, num_actions):
    flat_output, flat_output_size, network_parameters = create_conv_network(input_frames)
    fc1_W = tf.Variable(tf.random_normal([flat_output_size, 256], stddev=0.1), name='fc1_W')
    fc1_b = tf.Variable(tf.zeros([256]), name='fc1_b')
    # (batch size, 256)
    output3 = tf.nn.relu(tf.matmul(flat_output, fc1_W) + fc1_b, name='output3')

    fc2_W = tf.Variable(tf.random_normal([256, num_actions], stddev=0.1), name='fc2_W')
    fc2_b = tf.Variable(tf.zeros([num_actions]), name='fc2_b')
    # (batch size, num_actions)
    q_network = tf.nn.relu(tf.matmul(output3, fc2_W) + fc2_b, name='q_network')
    network_parameters +=  [fc1_W, fc1_b, fc2_W, fc2_b]

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
    outputV2 = tf.nn.relu(tf.matmul(outputV, fcV2_W) + fcV2_b, name='outputV2')


    fcA_W = tf.Variable(tf.random_normal([128, 512], stddev=0.1), name='fcA_W')
    fcA_b = tf.Variable(tf.zeros([512]), name='fcA_b')
    outputA = tf.nn.relu(tf.matmul(output1, fcA_W) + fcA_b, name='outputA')

    fcA2_W = tf.Variable(tf.random_normal([512, num_actions], stddev=0.1), name='fcA2_W')
    fcA2_b = tf.Variable(tf.zeros([num_actions]), name='fcA2_b')
    outputA2 = tf.nn.relu(tf.matmul(outputA, fcA2_W) + fcA2_b, name='outputA2')

    q_network = tf.nn.relu(outputV2 + outputA2 - tf.reduce_mean(outputA2), name='q_network')
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
    if question == 2:
        return {
            'replay_memory_size': batch_size,
            'target_update_freq': None,
            'create_network_fn': create_linear_q_network,
            'is_double_network': False,
            'is_double_dqn': False,
        }

    if question == 3:
        return {
            'replay_memory_size': 1000000,
            'target_update_freq': 10000,
            'create_network_fn': create_linear_q_network,
            'is_double_network': False,
            'is_double_dqn': False,
        }

    if question == 4:
        return {
            'replay_memory_size': 1000000,
            'target_update_freq': None,
            'create_network_fn': create_linear_q_network,
            'is_double_network': True,
            'is_double_dqn': False,
        }

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


def save_model(saver, sess, ai_input_dir, epsilon_generator):
    # Use a temp dir so nothing tries to read half-written files.
    temp_dir = tempfile.mkdtemp(prefix='melee-ai-manager')
    saver.save(sess, os.path.join(temp_dir, WORKER_INPUT_MODEL_FILENAME))
    with open(os.path.join(temp_dir, WORKER_INPUT_EPSILON_FILENAME), 'w') as epsilon_file:
        epsilon_file.write(str(epsilon_generator.get_epsilon(decay_epsilon=True)) + '\n')
    shutil.copy(WORKER_INPUT_RUN_SH_FILEPATH,
                os.path.join(temp_dir, os.path.basename(WORKER_INPUT_RUN_SH_FILEPATH)))

    shutil.move(temp_dir, os.path.join(ai_input_dir, str(time.time())))


def main():  # noqa: D103
    parser = argparse.ArgumentParser(description='Run DQN on Atari Space Invaders')
    parser.add_argument('--seed', default=10703, type=int, help='Random seed')
    parser.add_argument('--input_shape', default=SIZE_OF_STATE, help='Input shape')
    parser.add_argument('--gamma', default=0.99, help='Discount factor')
    # TODO experiment with this value.
    parser.add_argument('--epsilon', default=0.01, help='Final exploration probability in epsilon-greedy')
    parser.add_argument('--learning_rate', default=0.00025, help='Training learning rate.')
    parser.add_argument('--batch_size', default=500, type = int, help=
                                'Batch size of the training part')
    parser.add_argument('--question', type=int, default=7,
                        help='Which hw question to run.')


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


    # Copied from original phillip code (run.py).
    for opt in CPU.full_opts():
      opt.update_parser(parser)
    parser.add_argument("--dolphin", action="store_true", default=None, help="run dolphin")
    for opt in DolphinRunner.full_opts():
      opt.update_parser(parser)

    args = parser.parse_args()
    if args.generate_fixed_samples and args.is_manager:
        raise Exception('Can not generate fixed samples as manager. Must use ' +
                        '--is_worker and all other necessary flags (e.g. --cpu 9)')

    env = SmashEnv()
    if not args.is_manager:
        env.make(args)  # Opens Dolphin.

    question_settings = get_question_settings(args.question, args.batch_size)

    if args.is_manager:
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.set_random_seed(args.seed)

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


    saver = tf.train.Saver(max_to_keep=TOTAL_WORKER_JOBS)
    agent = DQNAgent(online_model=online_model,
                    target_model = target_model,
                    memory=replay_memory,
                    gamma=0.99,
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

        if args.is_manager:
            agent.compile(sess)
        else:
            saver.restore(sess, os.path.join(args.ai_input_dir, WORKER_INPUT_MODEL_FILENAME))

        print('_________________')
        print('number_actions: ' + str(env.action_space.n))

        # Worker code.
        if not args.is_manager:
          if random.random() < WORKER_EVALUATION_PROBABILITY:
              evaluation = agent.evaluate(env, sess, GreedyPolicy(), EVAL_EPISODES, MAX_EPISODE_LENGTH)
              print('Evaluation: ' + str(evaluation))
              with open(os.path.join(args.ai_output_dir, WORKER_OUTPUT_EVALUATE_FILENAME), 'wb') as f:
                  pickle.dump(evaluation, f)
              env.terminate()
              return

          with open(os.path.join(args.ai_input_dir, WORKER_INPUT_EPSILON_FILENAME)) as f:
              lines = f.readlines()
              # TODO handle unexpected lines better than just ignoring?
              worker_epsilon = float(lines[0])
              print('Worker epsilon: ' + str(worker_epsilon))
          train_policy = GreedyEpsilonPolicy(worker_epsilon)

          agent.play(env, sess, train_policy, num_iterations=NUM_WORKER_FRAMES, max_episode_length=MAX_EPISODE_LENGTH)
          replay_memory.save_to_file(os.path.join(args.ai_output_dir, WORKER_OUTPUT_GAMEPLAY_FILENAME))
          env.terminate()
          return



        # Manager code.
        print('Loading fix samples')
        with open(FIXED_SAMPLES_FILENAME, 'rb') as fixed_samples_f:
            fix_samples = pickle.load(fixed_samples_f)

        used_dirs = set()
        play_dirs = set()
        epsilon_generator = LinearDecayGreedyEpsilonPolicy(
            1.0, args.epsilon, TOTAL_WORKER_JOBS / 10.0)
        save_model(saver, sess, args.ai_input_dir, epsilon_generator)
        print('Begin to train (now safe to run gcloud)')
        print('Initial mean_max_q: ' + str(calculate_mean_max_Q(sess, online_model, fix_samples)))

        while len(play_dirs) < TOTAL_WORKER_JOBS:
            output_dirs = os.listdir(args.ai_output_dir)
            output_dirs = [os.path.join(args.ai_output_dir, x) for x in output_dirs]
            output_dirs = set(x for x in output_dirs if os.path.isdir(x))
            new_dirs = sorted(output_dirs - used_dirs)

            if len(new_dirs) == 0:
                time.sleep(0.1)
                continue

            new_dir = new_dirs[0]
            used_dirs.add(new_dir)
            evaluation_path = os.path.join(new_dir, WORKER_OUTPUT_EVALUATE_FILENAME)

            if os.path.isfile(evaluation_path):
                with open(evaluation_path, 'rb') as evaluation_file:
                    rewards, game_lengths = pickle.load(evaluation_file)
                mean_max_Q = calculate_mean_max_Q(sess, online_model, fix_samples)
                evaluation = [np.mean(rewards), np.std(rewards),
                              np.mean(game_lengths), np.std(game_lengths),
                              mean_max_Q]
                print('Evaluation: ' + '\t'.join(str(x) for x in evaluation))
                continue

            memory_path = os.path.join(new_dir, WORKER_OUTPUT_GAMEPLAY_FILENAME)
            if os.path.getsize(memory_path) == 0:
                # TODO Figure out why this happens despite temporary directory work.
                print('Output not ready somehow: ' + memory_path)
                time.sleep(0.1)
                continue

            with open(memory_path, 'rb') as memory_file:
                worker_memories = pickle.load(memory_file)
            for worker_memory in worker_memories:
                replay_memory.append(*worker_memory)


            play_dirs.add(new_dir)
            if len(play_dirs) <= NUM_BURN_IN_JOBS:
                print('Skip training because still burn in.')
                continue

            initial_step = (len(play_dirs) - NUM_BURN_IN_JOBS - 1) * FIT_PER_JOB
            for i in range(FIT_PER_JOB):
                agent.fit(sess, initial_step + i)

            # Partial evaluation to give frequent insight into agent progress.
            # Last time checked, this took ~0.1 seconds to complete.
            print('mean_max_q: ' + str(calculate_mean_max_Q(sess, online_model, fix_samples)))

            save_model(saver, sess, args.ai_input_dir, epsilon_generator)




if __name__ == '__main__':
    main()
