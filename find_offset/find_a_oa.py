import os
import inspect
import pickle

from matplotlib import pyplot as plt
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

TIMESTEPS_PER_ACTORBATCH = 4096
OPTIM_BATCHSIZE = 256
ENABLE_ENV_RANDOMIZER = True

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
      "act_fun": tf.nn.relu
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

def test(model, env, num_episodes=None):
    i = 0
    EPISODE = 600  # one episode has 600 step
    if num_episodes is None:
      collect_nums = 10000
    else:
      collect_nums = num_episodes
    oa_list = []
    oma_list = []
    o = env.reset()
    print(f"eposide: {i // EPISODE + 1} / {collect_nums}")
    while i < EPISODE * collect_nums:
        i += 1
        a, _ = model.predict(o, deterministic=True)
        print(a)
        o, r, done, info = env.step(a)
        oa = o[12:24]
        oa[np.array([1, 4, 7, 10])] -= 0.67
        oa[np.array([2, 5, 8, 11])] -= -1.25
        print(oa,'\n')
        oma = o[48:60]
        oa_list.append(oa)
        oma_list.append(oma)
        
        if done:
            o = env.reset()
            print(f"eposide: {i // EPISODE + 1} / {collect_nums}")
    env.close()
    
    # oa_list = np.array(oa_list)
    # oma_list = np.array(oma_list)
    # plt.figure()
    # for i in range(12):
    #     plt.subplot(4, 3, i+1)
    #     plt.plot(range(len(oa_list[:, i]),), oa_list[:, i], label=f'oa:{i}', linestyle='--')
    #     plt.plot(range(len(oma_list[:, i])), oma_list[:, i], label=f'oma:{i}', linestyle='-')
    #     plt.legend()
    # plt.show()   
    return

def main():
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("--seed", dest="seed", type=int, default=None)
  arg_parser.add_argument("--mode", dest="mode", type=str, default="test")
  arg_parser.add_argument("--motion_file", dest="motion_file", type=str, default="motion_imitation/data/motions/dog_pace.txt")
  arg_parser.add_argument("--visualize", dest="visualize", action="store_true", default=False)
  arg_parser.add_argument("--output_dir", dest="output_dir", type=str, default="output")
  arg_parser.add_argument("--num_test_episodes", dest="num_test_episodes", type=int, default=1)
  arg_parser.add_argument("--model_file", dest="model_file", type=str, default="motion_imitation/data/policies/dog_pace.zip")
  arg_parser.add_argument("--total_timesteps", dest="total_timesteps", type=int, default=2e8)
  arg_parser.add_argument("--int_save_freq", dest="int_save_freq", type=int, default=0) # save intermediate model every n policy steps

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


  if args.mode == "test":
      test(model=model,
           env=env,
           num_episodes=args.num_test_episodes)
  else:
      assert False, "Unsupported mode: " + args.mode

  return

if __name__ == '__main__':
  main()
