#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import glob
#import gym
from smash_env import SmashEnv
import numpy as np
import os
import random
import sys
import tensorflow as tf

from deeprl_hw2.core import ReplayMemory
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.objectives import mean_huber_loss
from deeprl_hw2.preprocessors import HistoryPreprocessor, PreprocessorSequence
from deeprl_hw2.policy import GreedyPolicy, LinearDecayGreedyEpsilonPolicy, UniformRandomPolicy
from deeprl_hw2.core import SIZE_OF_STATE

RMSP_EPSILON = 0.01
RMSP_DECAY = 0.95
RMSP_MOMENTUM =0.95
# TODO This used to be 20 (!). Consider increasing in future.
EVAL_EPISODES = 1
CHECKPOINT_EVAL_EPISODES = 100
MAX_EPISODE_LENGTH = 100000
NUM_FIXED_SAMPLES = 10000
# TODO Make this burnin and num fixed samples 50000 by restoring them from a file.
NUM_BURN_IN = 50
LINEAR_DECAY_LENGTH = 4000000


NUM_WORKER_FRAMES = 8 * 60 * 60 + 1000  # 1000 for just a little safety.



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
    W = tf.Variable(tf.random_normal([input_length, 128], stddev=0.01), name='W')
    b = tf.Variable(tf.zeros([128]), name='b')
    # (batch size, num_actions)
    output1 = tf.nn.relu(tf.matmul(input_frames_flat, W) + b, name='output1')

    fcV_W = tf.Variable(tf.random_normal([128, 512], stddev=0.01), name='fcV_W')
    fcV_b = tf.Variable(tf.zeros([512]), name='fcV_b')
    outputV = tf.nn.relu(tf.matmul(output1, fcV_W) + fcV_b, name='outputV')

    fcV2_W = tf.Variable(tf.random_normal([512, 1], stddev=0.01), name='fcV2_W')
    fcV2_b = tf.Variable(tf.zeros([1]), name='fcV2_b')
    outputV2 = tf.nn.relu(tf.matmul(outputV, fcV2_W) + fcV2_b, name='outputV2')


    fcA_W = tf.Variable(tf.random_normal([128, 512], stddev=0.01), name='fcA_W')
    fcA_b = tf.Variable(tf.zeros([512]), name='fcA_b')
    outputA = tf.nn.relu(tf.matmul(output1, fcA_W) + fcA_b, name='outputA')

    fcA2_W = tf.Variable(tf.random_normal([512, num_actions], stddev=0.01), name='fcA2_W')
    fcA2_b = tf.Variable(tf.zeros([num_actions]), name='fcA2_b')
    outputA2 = tf.nn.relu(tf.matmul(outputA, fcA2_W) + fcA2_b, name='outputA2')

    q_network = tf.nn.relu(outputV2 + outputA2 - tf.reduce_mean(outputA2), name='q_network')
    network_parameters = [W, b, fcV_W, fcV_b, fcV2_W, fcV2_b, fcA_W, fcA_b, fcA2_W, fcA2_b]
    return q_network, network_parameters



def create_model(window, input_shape, num_actions, model_name, create_network_fn, learning_rate):  # noqa: D103
    """Create the Q-network model."""
    with tf.name_scope(model_name):
        input_frames = tf.placeholder(tf.float32, [None, input_shape,
                        window], name ='input_frames')
        input_length = input_shape * window
        q_network, network_parameters = create_network_fn(
            input_frames, input_length, num_actions)

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


def main():  # noqa: D103
    parser = argparse.ArgumentParser(description='Run DQN on Atari Space Invaders')
    parser.add_argument('--env', default='SpaceInvadersDeterministic-v3', help='Atari env name')
    parser.add_argument('--seed', default=10703, type=int, help='Random seed')
    parser.add_argument('--input_shape', default=SIZE_OF_STATE, help='Input shape')
    parser.add_argument('--gamma', default=0.99, help='Discount factor')
    parser.add_argument('--epsilon', default=0.1, help='Exploration probability in epsilon-greedy')
    parser.add_argument('--learning_rate', default=0.00025, help='Training learning rate.')
    parser.add_argument('--window_size', default=4, type = int, help=
                                'Number of frames to feed to the Q-network')
    parser.add_argument('--batch_size', default=32, type = int, help=
                                'Batch size of the training part')
    parser.add_argument('--num_iteration', default=20000000, type = int, help=
                                'number of iterations to train')
    parser.add_argument('--eval_every', default=0.001, type = float, help=
                                'What fraction of num_iteration to run between evaluations.')
    parser.add_argument('--question', type=int, default=7,
                        help='Which hw question to run.')
    parser.add_argument('--eval_checkpoint_dir', type=str, default='',
                        help='Only evaluate each checkpoint in a given directory.')


    # TODO is this required?
    parser.add_argument('--ai_input_dir',
                        help='Input directory with initialization files.')
    parser.add_argument('--ai_output_dir',
                        help='Output directory for gameplay files.')
    parser.add_argument('--is_worker', dest='is_manager',
                        action='store_false',
                        help='Whether this is a worker (no training).')
    parser.add_argument('--is_manager', dest='is_manager',
                        action='store_true',
                        help='Whether this is a manager (trains).')
    parser.set_defaults(is_manager=True)
    env = SmashEnv()
    # TODO don't do this for is_manager.
    env.make(parser)

    args = parser.parse_args()
    question_settings = get_question_settings(args.question, args.batch_size)
    #env = gym.make(args.env)


    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    window_size = args.window_size
    online_model, online_params = create_model(
        window=window_size, input_shape=args.input_shape,
        num_actions=env.action_space.n, model_name='online_model',
        create_network_fn=question_settings['create_network_fn'],
        learning_rate=args.learning_rate)

    target_model = online_model
    update_target_params_ops = []
    if (question_settings['target_update_freq'] is not None or
        question_settings['is_double_network']):
        target_model, target_params = create_model(
            window=window_size, input_shape=args.input_shape,
            num_actions=env.action_space.n, model_name='target_model',
            create_network_fn=question_settings['create_network_fn'],
            learning_rate=args.learning_rate)
        update_target_params_ops = [t.assign(s) for s, t in zip(online_params, target_params)]

    # TODO load model from file for worker (not is_manager).
    # TODO load all other parameters from input too (e.g. epsilon).

    history_preprocessor = HistoryPreprocessor(history_length=window_size)
    preprocessor = PreprocessorSequence(history_preprocessor)

    memory = ReplayMemory(max_size=question_settings['replay_memory_size'],
                          window_length=window_size)

    policies = {
        'train_policy': LinearDecayGreedyEpsilonPolicy(1, args.epsilon, LINEAR_DECAY_LENGTH),
        'evaluate_policy': GreedyPolicy(),
    }

    saver = tf.train.Saver(max_to_keep=100000)
    agent = DQNAgent(online_model=online_model,
                    target_model = target_model,
                    preprocessor=preprocessor,
                    memory=memory, policies=policies,
                    gamma=0.99,
                    target_update_freq=question_settings['target_update_freq'],
                    update_target_params_ops=update_target_params_ops,
                    train_freq=4, batch_size=32,
                    is_double_network=question_settings['is_double_network'],
                    is_double_dqn=question_settings['is_double_dqn'])

    sess = tf.Session()
    fit_iterations = int(args.num_iteration * args.eval_every)
    checkpoint_iterations = [int(x * args.num_iteration) for x in [ 0.33, 0.66, 1]]
    max_eval_reward = -1.0

    with sess.as_default():
        if args.is_manager:
            agent.compile(sess)

        print('_________________')
        print('number_actions: ' + str(env.action_space.n))

        # TODO reenable?
        #print('Prepare fix samples and memory')
        #fix_samples = agent.prepare_fixed_samples(
        #    env, sess, UniformRandomPolicy(env.action_space.n),
        #    NUM_FIXED_SAMPLES, MAX_EPISODE_LENGTH)


        if not args.is_manager:
          # TODO do we need to limit by number of matches instead of number of frames?
          agent.fit(env, sess, num_iterations=NUM_WORKER_FRAMES, max_episode_length=NUM_WORKER_FRAMES, do_train=False)
          # TODO save ReplayMemory to output directory.
          return



        print('Prepare burn in')
        agent.fit(env, sess, num_iterations=NUM_BURN_IN, max_episode_length=MAX_EPISODE_LENGTH, do_train=False)
        print('Begin to train')
        for i in range(0, args.num_iteration, fit_iterations):

            if i in checkpoint_iterations:
                print('save tmp model'+str(i))
                saver.save(sess, 'tmp/model.%s.ckpt' % i)

            eval_reward, eval_stddev = agent.evaluate(env, sess, EVAL_EPISODES, MAX_EPISODE_LENGTH)
            #mean_max_Q = calculate_mean_max_Q(sess, online_model, fix_samples)
            info_string = str(i) + '\t' + str(eval_reward) + '\t' + str(eval_stddev) #+ '\t' + str(mean_max_Q)
            print(info_string)
            with open('tmpresult.txt', 'a') as myfile:
                myfile.write(info_string + '\n')

            if eval_reward > max_eval_reward:
                max_eval_reward = eval_reward
                print('save best yet model')
                saver.save(sess, 'tmp/model.%s.%s.ckpt' % (i, eval_reward))

            agent.fit(env, sess, start_iteration=i, num_iterations=fit_iterations, max_episode_length=MAX_EPISODE_LENGTH)
            sys.stdout.flush()


        print('______Final Results________')
        saver.save(sess, 'tmp/model.%s.ckpt' % args.num_iteration)
        print('Model saved in file model.%s.ckpt' % args.num_iteration)

        eval_reward, eval_stddev = agent.evaluate(env, sess, EVAL_EPISODES, MAX_EPISODE_LENGTH)
        info_string = 'Final\t' + str(eval_reward) + '\t' + str(eval_stddev)
        print(info_string)
        with open('tmpresult.txt', 'a') as myfile:
            myfile.write(info_string + '\n')



if __name__ == '__main__':
    main()
