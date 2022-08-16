import os
import reverb

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment

from tf_agents.networks import q_network
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.policies import policy_saver
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.train.utils import train_utils

from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4

from pandas import DataFrame


def run_fNIRS():
  # ## Hyperparameters

  num_iterations = 200000 # @param {type:"integer"}

  initial_collect_steps = 100  # @param {type:"integer"}
  collect_steps_per_iteration =   1# @param {type:"integer"}
  replay_buffer_max_length = 100000  # @param {type:"integer"}

  batch_size = 64  # @param {type:"integer"}
  learning_rate = 25e-5  # @param {type:"number"}
  log_interval = 200  # @param {type:"integer"}

  num_eval_episodes = 10  # @param {type:"integer"}
  eval_interval = 1000  # @param {type:"integer"}

  # Load the PacMan environment from the OpenAI Gym suite. 

  env_name = 'MsPacman-v0'
  # env_name = 'ALE/MsPacman-v5'

  #initial environment
  env = suite_gym.load(env_name, 
                      gym_env_wrappers=[AtariPreprocessing, FrameStack4])

  print('Observation Spec:')
  print(env.time_step_spec().observation)

  print('Reward Spec:')
  print(env.time_step_spec().reward)

  print('Action Spec:')
  print(env.action_spec())
  env.unwrapped.get_action_meanings()

  # In the PacMan environment:
  # 
  # -   `observation` is an RGB picture of size $210\times160$. Each frame represents: 
  #     -   the position of the PacMan and ghosts
  #     -   the position of rewards and goals 
  # -   `reward` is a scalar float value
  # -   `action` is a scalar integer with only 9 possible values:
  #     -   `0` — "no move"
  #     -   `1` — "move right"
  #     -   `2` — "move right"
  #     -   `3` — "move left"
  #     -   `4` — "move down"
  #     -   `5` — "move upright"
  #     -   `6` — "move upleft"
  #     -   `7` — "move downright"
  #     -   `8` — "move downleft"


  # Usually two environments are instantiated: one for training and one for evaluation. 
  train_py_env = suite_gym.load(env_name, 
                      gym_env_wrappers=[AtariPreprocessing, FrameStack4])
  eval_py_env = suite_gym.load(env_name, 
                      gym_env_wrappers=[AtariPreprocessing, FrameStack4])

  train_env = tf_py_environment.TFPyEnvironment(train_py_env)
  eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

  ## Agent
  # At the heart of a DQN Agent is a `QNetwork`, a neural network model that can learn 
  # to predict `QValues` (expected returns) for all actions, given an observation from the environment.
  # The `QNetwork` will consist of a sequence of `tf.keras.layers.Dense` layers, 
  # where the final layer will have 1 output for each possible action.
  fc_layer_params = [512]
  conv_layer_params = [(32, (8, 8), 4), 
                      (64, (4, 4), 2), 
                      (64, (3, 3), 1)
                      ]
  q_net = q_network.QNetwork(
      train_env.observation_spec(),
      train_env.action_spec(),
      preprocessing_layers=tf.keras.layers.Lambda(lambda x: tf.cast(x, dtype=tf.float32)),
      conv_layer_params=conv_layer_params,
      fc_layer_params=fc_layer_params,
      name='QNetwork')
  update_period = 4
  epsilon_fn  = tf.optimizers.schedules.PolynomialDecay(
                    initial_learning_rate = 0.1,
                    decay_steps = 250000//update_period,
                    end_learning_rate = 0.01
                    )

  optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
  global_step = train_utils.create_train_step()

  # Now use `tf_agents.agents.dqn.dqn_agent` to instantiate a `DqnAgent`. 
  # In addition to the `time_step_spec`, `action_spec` and the QNetwork, 
  # the agent constructor also requires an optimizer (in this case, `AdamOptimizer`), 
  # a loss function, and an integer step counter.
  agent = dqn_agent.DqnAgent(
      train_env.time_step_spec(),
      train_env.action_spec(),
      q_network=q_net,
      optimizer=optimizer,
      target_update_period=2000,
      td_errors_loss_fn=tf.keras.losses.Huber(reduction='none'),
      gamma=0.99,
      train_step_counter=global_step,
      epsilon_greedy=lambda:epsilon_fn(global_step),
      gradient_clipping=None,
      summarize_grads_and_vars=True,
      debug_summaries=True)

  # initilize the agent
  agent.initialize()

  ## Policies
  # 
  # We use two independent policies: 
  # 
  # -   `agent.policy` — The main policy that is used for evaluation and deployment.
  # -   `agent.collect_policy` — A second policy that is used for data collection.
  eval_policy = agent.policy
  collect_policy = agent.collect_policy

  # Then we  use a `tf_agents.policies.random_tf_policy` to create a policy which 
  # will randomly select an action for each `time_step` as baseline.
  random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                  train_env.action_spec())

  # To get an action from a policy, call the `policy.action(time_step)` method. 
  # The `time_step` contains the observation from the environment. This method 
  # returns a `PolicyStep`, which is a named tuple with three components:
  # 
  # -   `action` — the action to be taken (in this case, `0` or `8`)
  # -   `state` — which represents observations from the environment
  # -   `info` — auxiliary data, such as log probabilities of actions

  ## random policy as baseline
  example_environment = tf_py_environment.TFPyEnvironment(
      suite_gym.load(env_name, 
                      gym_env_wrappers=[AtariPreprocessing, FrameStack4]))

  time_step = example_environment.reset()
  random_policy.action(time_step)

  ## Metrics and Evaluation
  # The most common metric used to evaluate a policy is the average return. 
  # The following function computes the average return of a policy, given 
  def compute_avg_return(environment, policy, num_episodes=10):

    total_return = 0.0
    for _ in range(num_episodes):

      time_step = environment.reset()
      episode_return = 0.0

      while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = environment.step(action_step.action)
        episode_return += time_step.reward
      total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


  # Running this metric on the `random_policy` shows a 
  # baseline performance in the environment.
  compute_avg_return(eval_env, random_policy, num_eval_episodes)



  ## Replay Buffer
  # In order to keep track of the data collected from the environment, 
  # we will use a replay system called Reverb. It stores experience data 
  # when we collect trajectories and is consumed during training.

  # Here, for the sake of this experiment we use a uniform table, meaning 
  # that data stored in the replay buffer is sampled uniformly. Alternatively,
  # we can use prioritized buffer to use some samples (typically human demonstrations)
  # more likely than others (typically the agent's explorations)
  table_name = 'uniform_table'
  replay_buffer_signature = tensor_spec.from_spec(
        agent.collect_data_spec)
  replay_buffer_signature = tensor_spec.add_outer_dim(
      replay_buffer_signature)

  table = reverb.Table(
      table_name,
      max_size=replay_buffer_max_length,
      sampler=reverb.selectors.Uniform(),
      remover=reverb.selectors.Fifo(),
      rate_limiter=reverb.rate_limiters.MinSize(1),
      signature=replay_buffer_signature)

  reverb_server = reverb.Server([table])

  replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
      agent.collect_data_spec,
      table_name=table_name,
      sequence_length=2,
      local_server=reverb_server)

  rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
    replay_buffer.py_client,
    table_name,
    sequence_length=2)


  ## setup checkpointer and policy saver
  cwd = os.getcwd()
  checkpoint_dir = os.path.join(cwd, 'checkpoint')
  train_checkpointer = common.Checkpointer(
      ckpt_dir=checkpoint_dir,
      max_to_keep=1,
      agent=agent,
      policy=agent.policy,
      replay_buffer=replay_buffer,
      global_step=global_step
  )

  policy_dir = os.path.join(cwd, 'policy')
  tf_policy_saver = policy_saver.PolicySaver(agent.policy)


  ## Data Collection
  # We begin by executing the random policy in the environment for a few steps, 
  # recording the data in the replay buffer.
  # The replay buffer is now a collection of Trajectories collected for some random actions
  py_driver.PyDriver(
      env,
      py_tf_eager_policy.PyTFEagerPolicy(
        random_policy, use_tf_function=True),
      [rb_observer],
      max_steps=initial_collect_steps).run(train_py_env.reset());

  # Each row of the replay buffer only stores a single observation step. But since 
  # the DQN Agent needs both the current and next observation to compute the loss, 
  # the dataset pipeline will sample two adjacent rows for each item in the batch (`num_steps=2`).
  # Dataset generates trajectories with shape [Bx2x...]
  dataset = replay_buffer.as_dataset(
      num_parallel_calls=3,
      sample_batch_size=batch_size,
      num_steps=2).prefetch(3)

  iterator = iter(dataset)

  ## Training the agent
  # During the training process, we periodicially (every 1000 steps) evaluates 
  # the policy and prints the current score (by computing the metrics).
  # agent.train = common.function(agent.train)

  # Reset the train step.
  agent.train_step_counter.assign(0)

  # Evaluate the agent's policy once before training to collect the baseline performance.
  avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
  returns = [avg_return]

  # Reset the environment.
  time_step = train_py_env.reset()

  # Create a driver to collect experience.
  collect_driver = py_driver.PyDriver(
      env,
      py_tf_eager_policy.PyTFEagerPolicy(
        agent.collect_policy, use_tf_function=True),
      [rb_observer],
      max_steps=collect_steps_per_iteration)

  for episode in range(num_iterations):
    # Collect a few steps and save to the replay buffer.
    time_step, _ = collect_driver.run(time_step)

    # Sample a batch of data from the buffer and update the agent's network.
    experience, _ = next(iterator)
    train_loss = agent.train(experience).loss

    step = agent.train_step_counter.numpy()

    if step % log_interval == 0:
      print('step = {0}: loss = {1}'.format(step, train_loss))

    if step % eval_interval == 0:
      avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
      print('step = {0}: Average Return = {1}'.format(step, avg_return))
      returns.append(avg_return)


  # At the end of the process, save the policies and checkpointers

  # train_checkpointer.save(global_step)
  # tf_policy_saver.save(policy_dir)


  def save_data(df):
      def file_name(tag:int)->str:
          return os.path.join(cwd,'output/') + \
          'results_iter{:0=3d}.csv'.format(tag)
      i = 1
      while os.path.exists(file_name(i)):
          i += 1
      df.to_csv(file_name(i))

  ## Plots
  # Plotting the average return to monitor the progress of learning.
  iterations = range(0, num_iterations + 1, eval_interval)
  data = {"iters":list(iterations), "returns": returns}
  df = DataFrame(data, columns=["iters", "returns"])
  save_data(df)
  # plt.plot(df["iters"], df["returns"])
  # plt.ylabel('Average Return')
  # plt.xlabel('Iterations')
  # plt.ylim(top=2000)

  # to render videos of showing how the agent is learning to act, use the 
  # provided jupyter notebook.


if __name__ == '__main__':
  for rep in range(3):
    run_fNIRS()
    
