# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os
import inspect
import pickle
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import argparse
from mpi4py import MPI
import numpy as np
import os
import random
import tensorflow as tf
import time

from motion_imitation.envs import env_builder as env_builder
from motion_imitation.learning import imitation_policies as imitation_policies
from motion_imitation.learning import ppo_imitation as ppo_imitation

from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines.gail.dataset.dataset import ExpertDataset


NOWTIME = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

TIMESTEPS_PER_ACTORBATCH = 4096
OPTIM_BATCHSIZE = 256
ENABLE_ENV_RANDOMIZER = True

TIMESTEP = 1 / 30
max_kp = 600

def pma_to_oma(pma):
    oma = copy.deepcopy(pma)
    oma[:, np.array([0, 6])] = -pma[:, np.array([0, 6])]
    oma[:, np.array([0, 6])] += 0.30
    oma[:, np.array([3, 9])] -= 0.30
    oma[:, np.array([1, 4, 7, 10])] += 0.6
    oma[:, np.array([2, 5, 8, 11])] += -0.66
    return oma
def oma_to_right_action(oma):
    action = copy.deepcopy(oma)
    # action[:, np.array([0, 6])] = -oma[:, np.array([0, 6])]
    action[:, np.array([0, 3, 6, 9])] = -oma[:, np.array([0, 3, 6, 9])]
    action[:, np.array([1, 4, 7, 10])] -= 0.67
    action[:, np.array([2, 5, 8, 11])] -= -1.25
    return action
def set_rand_seed(seed=None):
  if seed is None:
    seed = int(time.time())

  seed += 97 * MPI.COMM_WORLD.Get_rank()

  tf.set_random_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

  return

def build_model(env, num_procs, timesteps_per_actorbatch, optim_batchsize, output_dir):
  policy_kwargs = {
      "net_arch": [{"pi": [512, 256],
                    "vf": [512, 256]}],
      "act_fun": tf.nn.leaky_relu
  }

  timesteps_per_actorbatch = int(np.ceil(float(timesteps_per_actorbatch) / num_procs))
  optim_batchsize = int(np.ceil(float(optim_batchsize) / num_procs))

  model = ppo_imitation.PPOImitation(
               policy=imitation_policies.ImitationPolicy,
               env=env,
               gamma=0.95,
               timesteps_per_actorbatch=timesteps_per_actorbatch,
               clip_param=0.2,
               optim_epochs=1,
               optim_stepsize=1e-5,
               optim_batchsize=optim_batchsize,
               lam=0.95,
               adam_epsilon=1e-5,
               schedule='constant',
               policy_kwargs=policy_kwargs,
               tensorboard_log=output_dir,
               verbose=1)
  return model


def train(model, env, total_timesteps, output_dir="", int_save_freq=0, pretrain_flag=False, output_is_speed=False, load_model_flag=False, file_path = f'output/pretrained_model/pretrained_model_100_steps_{NOWTIME}'):
  if (output_dir == ""):
    save_path = None
  else:
    save_path = os.path.join(output_dir, "model_8_14_load_2e8.zip")
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
  

  callbacks = []
  # Save a checkpoint every n steps
  if (output_dir != ""):
    if (int_save_freq > 0):

      int_dir = os.path.join(output_dir, "intermedate")
      callbacks.append(CheckpointCallback(save_freq=int_save_freq, save_path=int_dir,
                                          name_prefix=f'model_{NOWTIME}'))
  if pretrain_flag:
    with open('dataset/augmentation_trot_data_1000_100.pkl', 'rb') as f:
      file_data = f.read()
      
      allresult = MPI.pickle.loads(file_data)
    preobs = allresult['input'][:,:]  # point
    actions = allresult['output'][:,:]  # velocity
    
    
    """
    第一种是在预训练数据里面加偏置，让预训练后的模型喜欢岔开腿走路，
    第二种是预训练里面不加偏执，但是在正式训练的时候在observation喂给模型的时候假装腿往中间收了，模型的输出乘以时间再加上原始的observation作为action
    
    # 7.23 第一种方法，训练数据加偏置
    actions *= 1 / 30  # displacement
    next_obs = actions + preobs
    
    # pma to oma
    next_obs_o_space = copy.deepcopy(next_obs)
    next_obs_o_space[:, np.array([0, 6])] = -next_obs[:, np.array([0, 6])]
    next_obs_o_space[:, np.array([1, 4, 7, 10])] += -0.6
    next_obs_o_space[:, np.array([2, 5, 8, 11])] += 0.66
    
    # oma to action
    next_obs_a_space = copy.deepcopy(next_obs_o_space)
    next_obs_a_space[:, np.array([0, 3, 6, 9])] = -next_obs_o_space[:, np.array([0, 3, 6, 9])]
    next_obs_a_space[:, np.array([1, 4, 7, 10])] -= 0.67
    next_obs_a_space[:, np.array([2, 5, 8, 11])] -= -1.25
    """
    # 8.09 修正偏置
    pma = preobs
    velocity = actions
    oma = pma_to_oma(pma)
    next_oma = oma + velocity * TIMESTEP
    action = oma_to_right_action(next_oma)
    
    del allresult
    obs = np.zeros((len(preobs),160))
    
    obs[:, 48:60] = oma
    # create the dataset
    
    traj_data = {'obs': obs,
                # 'actions': actions * 1 / 30 + oma_to_right_action(preobs),
                'actions': action,
                'episode_returns': np.zeros(len(preobs), dtype=bool),
                'rewards': np.zeros(len(preobs), dtype=bool),
                'episode_starts': np.zeros(len(preobs), dtype=bool)}
    # 'actions', 'episode_returns', 'rewards', 'obs' and 'episode_starts'
    dataset = ExpertDataset(traj_data=traj_data, )
    model.pretrain(dataset, n_epochs=100, learning_rate=1e-4, adam_epsilon=1e-8,)
    # path = os.path.join('output/pretrained_model/', '{}_{}_steps'.format('pretrained_model', 1))
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    model.save(file_path)
  # return
  # o = env.reset()
  # a1, _ = model.predict(o, deterministic=True)
  # print('no load a :', a1)
  if load_model_flag:
    # file_path = f'output/pretrained_model/pretrained_model_100_steps_2024_08_09_12_18_52.zip'
    file_path=f'output/pretrained_model/pretrained_model_100_steps_2024_08_09_16_05_39.zip'
    
    # file_path=f'E:\VScode\motion-imitation\motion_imitation\data\policies\dog_trot.zip'
    model.load_parameters(file_path)
    # a2, _ = model.predict(o, deterministic=True)
    
    # print('load a :', a2)
    # while True:
    #   o, r, done, info = env.step(a2)
    #   a2, _ = model.predict(o, deterministic=True)
    #   if done:
    #     o = env.reset()
    
  model.learn(total_timesteps=total_timesteps, save_path=save_path, callback=callbacks, output_is_speed=output_is_speed)

  return

def test(model, env, num_procs, num_episodes=None, output_is_speed=False):
  curr_return = 0
  sum_return = 0
  episode_count = 0

  if num_episodes is not None:
    num_local_episodes = int(np.ceil(float(num_episodes) / num_procs))
  else:
    num_local_episodes = np.inf

  o = env.reset()
  while episode_count < num_local_episodes:
    a, _ = model.predict(o, deterministic=True)
    if output_is_speed:
      env_action = a * TIMESTEP + o[48:60]
      env_action = oma_to_right_action(env_action)
      a = env_action
    o, r, done, info = env.step(a)
    curr_return += r

    if done:
        o = env.reset()
        sum_return += curr_return
        episode_count += 1

  sum_return = MPI.COMM_WORLD.allreduce(sum_return, MPI.SUM)
  episode_count = MPI.COMM_WORLD.allreduce(episode_count, MPI.SUM)

  mean_return = sum_return / episode_count

  if MPI.COMM_WORLD.Get_rank() == 0:
      print("Mean Return: " + str(mean_return))
      print("Episode Count: " + str(episode_count))

  return

def main():
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("--seed", dest="seed", type=int, default=0)
  arg_parser.add_argument("--mode", dest="mode", type=str, default="train")
  arg_parser.add_argument("--motion_file", dest="motion_file", type=str, default="motion_imitation/data/motions/dog_trot.txt")
  arg_parser.add_argument("--visualize", dest="visualize", action="store_true", default=False)
  arg_parser.add_argument("--output_dir", dest="output_dir", type=str, default="output")
  arg_parser.add_argument("--num_test_episodes", dest="num_test_episodes", type=int, default=None)
  arg_parser.add_argument("--model_file", dest="model_file", type=str, default="")
  arg_parser.add_argument("--total_timesteps", dest="total_timesteps", type=int, default=2e8)
  arg_parser.add_argument("--int_save_freq", dest="int_save_freq", type=int, default=1e7) # save intermediate model every n policy steps
  arg_parser.add_argument("--pretrain_flag", dest="pretrain_flag", type=bool, default=False)
  arg_parser.add_argument("--output_is_speed", dest="output_is_speed", type=bool, default=False)
  arg_parser.add_argument("--load_model_flag", dest="load_model_flag", type=bool, default=False)

  args = arg_parser.parse_args()
  
  num_procs = MPI.COMM_WORLD.Get_size()
  os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
  
  enable_env_rand = ENABLE_ENV_RANDOMIZER and (args.mode != "test")
  env = env_builder.build_imitation_env(motion_files=[args.motion_file],
                                        num_parallel_envs=num_procs,
                                        mode=args.mode,
                                        enable_randomizer=enable_env_rand,
                                        enable_rendering=args.visualize)
  
  model = build_model(env=env,
                      num_procs=num_procs,
                      timesteps_per_actorbatch=TIMESTEPS_PER_ACTORBATCH,
                      optim_batchsize=OPTIM_BATCHSIZE,
                      output_dir=args.output_dir)

  
  if args.model_file != "":
    model.load_parameters(args.model_file)
  print('pretrain_flag:',args.pretrain_flag)
  print('load_model_flag:',args.load_model_flag)
  if args.mode == "train":
      train(model=model, 
            env=env, 
            total_timesteps=args.total_timesteps,
            output_dir=args.output_dir,
            int_save_freq=args.int_save_freq,
            pretrain_flag=args.pretrain_flag,
            output_is_speed=args.output_is_speed,
            load_model_flag=args.load_model_flag)
  elif args.mode == "test":
      test(model=model,
           env=env,
           num_procs=num_procs,
           num_episodes=args.num_test_episodes,
           output_is_speed=args.output_is_speed)
  else:
      assert False, "Unsupported mode: " + args.mode

  return

if __name__ == '__main__':
  main()
