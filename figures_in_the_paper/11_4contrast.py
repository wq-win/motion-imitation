import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import copy
import pickle
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from motion_imitation.envs import env_builder
from motion_imitation.learning import imitation_policies as imitation_policies
from motion_imitation.learning import ppo_imitation as ppo_imitation
from mpi4py import MPI

# training set a fixed parameter ?
# pretrian_flag = False

before_pretrain_action = []
before_pretrain_obs_action = []
after_pretrain_action = []
after_pretrain_obs_action = []


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

def before_pretrain():
    pass

def after_pretrain(model, env, num_episodes=None):
    o = env.reset()
    for i in range(num_episodes):
        a, _ = model.predict(o, deterministic=True)
        o, r, d, info = env.step(a)
        if d:
            o = env.reset()

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    env = env_builder.build_imitation_env(
        motion_files=["motion_imitation/data/motions/dog_trot.txt"],
        num_parallel_envs=MPI.COMM_WORLD.Get_size(),
        mode="test",
        enable_randomizer=False,
        enable_rendering=True,
        if_trajectory_generator=True)
    
    before_pretrain()
    
    # before pretrain, use expert data


    # after pretrain, use model
    model = build_model(
        env=env,
        num_procs=MPI.COMM_WORLD.Get_size(),
        timesteps_per_actorbatch=4096,
        optim_batchsize=256,
        output_dir='output')
    
    model.load_parameters("output/ppo_imitation_model_final.pkl")
    # add pretrain logic here
    after_pretrain(model, env, num_episodes=1000)
if __name__ == "__main__":
    main()