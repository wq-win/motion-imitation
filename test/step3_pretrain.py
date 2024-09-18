import os
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

from stable_baselines.gail.dataset.dataset import ExpertDataset
from stable_baselines.common.callbacks import CheckpointCallback
from motion_imitation.learning import ppo_imitation as ppo_imitation
from motion_imitation.learning import imitation_policies as imitation_policies
from motion_imitation.envs import env_builder as env_builder
import time
import tensorflow as tf
import random
import numpy as np
from mpi4py import MPI
import argparse
import copy
import pickle


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

    timesteps_per_actorbatch = int(
        np.ceil(float(timesteps_per_actorbatch) / num_procs))
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


def train(model, env, total_timesteps, output_dir="", int_save_freq=0):
    save_path = os.path.join(output_dir, "model_9_10_right_expertdata.zip")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    callbacks = []
    # Save a checkpoint every n steps
    if (output_dir != ""):
        if (int_save_freq > 0):
            int_dir = os.path.join(output_dir, "intermedate")
            callbacks.append(CheckpointCallback(save_freq=int_save_freq, save_path=int_dir,
                                                name_prefix=f'model_{NOWTIME}'))
    
    with open('dataset/expertdata_10_600.pkl', 'rb') as f:
        file_data = f.read()
        traj_data = MPI.pickle.loads(file_data)

        dataset = ExpertDataset(traj_data=traj_data, )
        model.pretrain(dataset, n_epochs=100, learning_rate=1e-4, adam_epsilon=1e-8,)
    print('start learn')    
    model.learn(total_timesteps=total_timesteps, save_path=save_path,
                callback=callbacks, output_is_speed=False)
    print('over learn')  
    return
           
    #     save_file_path=f'output/expertdata_model/expertdata_10.zip'
    #     directory = os.path.dirname(save_file_path)
    #     if not os.path.exists(directory):
    #         os.makedirs(directory)
    #     model.save(save_file_path)
    # return
    

def main():
    num_procs = MPI.COMM_WORLD.Get_size()
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

    env = env_builder.build_imitation_env(motion_files=[os.path.join(parentdir,"motion_imitation/data/motions/dog_trot.txt")],
                                    num_parallel_envs=MPI.COMM_WORLD.Get_size(),
                                    mode="train",
                                    enable_randomizer=True,
                                    enable_rendering=True,
                                    if_trajectory_generator=True)

    model = build_model(
        env=env,
        num_procs=num_procs,
        timesteps_per_actorbatch=TIMESTEPS_PER_ACTORBATCH,
        optim_batchsize=OPTIM_BATCHSIZE,
        output_dir="output",
    )

    train(model=model,
            env=env,
            total_timesteps=2e8,
            output_dir="output",
            int_save_freq=1e7,
            )

    return


if __name__ == '__main__':
    main()
