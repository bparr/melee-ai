"""Main DQN agent."""
import numpy as np
import tensorflow as tf
import random
import time
from .core import mprint


# Number of frames each select_action should be used for.
FRAMES_PER_ACTION = 1

TOTAL_TIME = 0.0


def _run_episode(env, max_episode_length, select_action_fn, process_step_fn, start_step=0, end_seconds=None):
    """Step through a single game episode.

    NOTE: This will reset env!
    select_action_fn takes in the state, and returns the selected action, q_values
    process_step_fn takes in old_state, reward, action, new_state, is_terminal, q_values

    end_seconds is the time.time() that should stop running an episode.

    Returns
    --------
    The new step number (start_step + number of steps taken).
    """
    state = env.reset()
    for current_step in range(start_step, start_step + max_episode_length):
        action, q_values = select_action_fn(state)
        old_state = state
        reward = 0.0
        is_terminal = False
        for i in range(FRAMES_PER_ACTION):
            state, intermediate_reward, is_terminal, env_done = env.step(action)
            reward += intermediate_reward
            if env_done:
                break

        process_step_fn(old_state, reward, action, state, is_terminal, q_values)
        if env_done:
          return current_step + 1

        # TODO if FRAMES_PER_ACTION > 1 then we could end up doing
        # (FRAMES_PER_ACTION - 1) too many actions before timeout.
        if end_seconds is not None and time.time() > end_seconds:
          return current_step + 1

    return start_step + max_episode_length


class DQNAgent:
    """Class implementing DQN.

    Parameters
    ----------
    online_model: tf.Tensor
    target_model: tf.Tensor
    memory: deeprl_hw2.core.Memory
    gamma: float
      Discount factor.
    target_update_freq: float
      Frequency to update the target network. You can either provide a
      number representing a soft target update (see utils.py) or a
      hard target update (see utils.py and Atari paper.)
    update_target_params_ops: list of tf.Tensor
      List of tensorflow assign operators that update the target_model
      to have the same weights as the online model.
    batch_size: int
      How many samples in each minibatch.
    is_double_network: boolean
      Whether to treat the online/target models as a double network
      (coin flipping).
    is_double_dqn: boolean
      Whether to treat the online/target models as a double dqn.
    """
    def __init__(self,
                 online_model,
                 target_model,
                 memory,
                 gamma,
                 target_update_freq,
                 update_target_params_ops,
                 batch_size,
                 is_double_network,
                 is_double_dqn):
        self._online_model = online_model
        self._target_model = target_model
        self._memory = memory
        self._gamma = gamma
        self._target_update_freq = target_update_freq
        self._update_target_params_ops = update_target_params_ops
        self._batch_size = batch_size
        self._is_double_network = is_double_network
        self._is_double_dqn = is_double_dqn


    def compile(self, sess):
        """Setup all of the TF graph variables/ops."""
        sess.run(tf.global_variables_initializer())

        if self._target_update_freq is not None:
            sess.run(self._update_target_params_ops)



    def calc_q_values(self, sess, state, model):
        """Given a state (or batch of states) calculate the Q-values.

        Return
        ------
        Q-values for the state(s)
        """
        feed_dict = {model['input_frames']: state}
        q_values = sess.run(model['q_network'], feed_dict=feed_dict)
        return q_values


    def select_action(self, sess, state, policy, model):
        """Select the action based on the current state.

        Returns
        --------
        selected action
        """
        q_values = self.calc_q_values(sess, state, model)
        #print(q_values, q_values[0][1] - q_values[0][0])
        return policy.select_action(q_values=q_values), tuple(q_values[0])


    def play(self, env, sess, policy, total_seconds,
             start_iteration=0, max_episode_length=1):
        """Play the game, no training.

        Parameters
        ----------
        env: gym.Env
          This is your Atari environment. You should wrap the
          environment using the wrap_atari_env function in the
          utils.py
        sess: tf.Session
        policy: policy.Policy
        total_seconds: int
          Total number of real seconds to play.
        start_iteration: int
          Starting number for iteration counting. Useful when calling fit
          multiple times.
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        """
        def select_action_fn(state):
            return self.select_action(sess, state, policy, self._online_model)

        def process_step_fn(old_state, reward, action, state, is_terminal, q_values):
            self._memory.append(old_state, reward, action, state, is_terminal, q_values)


        end_seconds= time.time() + total_seconds
        episode = 0
        print('Playing for ' + str(total_seconds) + ' seconds.')
        while time.time() < end_seconds:
            episode += 1
            print('Running episode: ' + str(episode))
            _run_episode(env, max_episode_length,
                select_action_fn, process_step_fn, end_seconds=end_seconds)


    def fit(self, sess, current_step):
        """Fit your model to the provided environment.

        Its a good idea to print out things like loss, average reward,
        Q-values, etc to see if your agent is actually improving.

        You should probably also periodically save your network
        weights and any other useful info.

        This is where you should sample actions from your network,
        collect experience samples and add them to your replay memory,
        and update your network parameters.

        Parameters
        ----------
        sess: tf.Session
        current_step: How many steps of fit we have done so far.
        """
        # TODO remove later since only for double linear q network?
        model1 = self._online_model
        model2 = self._target_model
        if self._is_double_network and random.random() < 0.5:
            model1, model2 = model2, model1

        # Get sample
        old_state_list, reward_list, action_list, new_state_list, is_terminal_list, _ = self._memory.sample(self._batch_size)

        # calculate y_j
        Q_values = self.calc_q_values(sess, new_state_list, model2)
        if self._is_double_dqn:
            target_action_list = self.calc_q_values(
                sess, new_state_list, model1).argmax(axis=1)
            max_q = [Q_values[i, j] for i, j in enumerate(target_action_list)]
        else:
            max_q = Q_values.max(axis=1)
        # Improve network stability by clipping rewards.
        y = np.clip(reward_list, -1.0, 1.0)
        for i in range(len(is_terminal_list)):
          if not is_terminal_list[i]:
              y[i] += self._gamma * max_q[i]


        # Train on memory sample.
        #feed_dict = {model1['input_frames']: old_state_list,
        #             model1['Q_vector_indexes']: list(enumerate(action_list)),
        #             model1['y_ph']: y}


        start_time = time.time()
        #sess.run([model1['train_step']], feed_dict=feed_dict)
        sess.run([model1['train_step']])


        global TOTAL_TIME
        TOTAL_TIME += time.time() - start_time
        if (self._target_update_freq is not None and
            current_step % self._target_update_freq == 0):
            mprint('Updating target network')
            sess.run(self._update_target_params_ops)


    def print_total_time(self):
      global TOTAL_TIME
      print(TOTAL_TIME)

    def evaluate(self, env, sess, policy, num_episodes, max_episode_length):
        """Test your agent with a provided environment."""
        rewards = []
        game_lengths = []

        def select_action_fn(state):
            return self.select_action(sess, state, policy, self._online_model)

        def process_step_fn(old_state, reward, action, state, is_terminal, q_values):
            rewards[-1] += reward

        for episode in range(num_episodes):
            rewards.append(0.0)
            _run_episode(env, max_episode_length,
                         select_action_fn, process_step_fn)
            game_lengths.append(env.get_game_length())
            print('Game: ', episode, rewards[-1], game_lengths[-1])

        return rewards, game_lengths


    def prepare_fixed_samples(self, env, sess, policy, num_samples, max_episode_length):
        """Return a set of fixed samples reached using the given policy.

        Returns
        --------
        List of states stored as floats and so can be directly fed into a
        tensorflow model.
        """
        samples = []

        def select_action_fn(state):
            return self.select_action(sess, state, policy, self._online_model)

        def process_step_fn(old_state, reward, action, state, is_terminal, q_values):
            if state.shape[0] != 1:
                raise Exception('Unexpected state shape in prepare_fixed_samples')
            samples.append(state[0])


        while len(samples) < num_samples:
            print(len(samples))
            _run_episode(env,
                         min(max_episode_length, num_samples - len(samples)),
                         select_action_fn, process_step_fn)

        return samples

