import copy
from matplotlib import pyplot as plt
import torch
import augmentation_data
import data
import os
import inspect
import pickle
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
from find_offset import all_offset
from stable_baselines.common.callbacks import CheckpointCallback
from motion_imitation.learning import ppo_imitation as ppo_imitation
from motion_imitation.learning import imitation_policies as imitation_policies
from motion_imitation.envs import env_builder as env_builder
import tensorflow as tf
import numpy as np
from mpi4py import MPI


TIMESTEPS_PER_ACTORBATCH = 4096
OPTIM_BATCHSIZE = 256
TIMESTEP = augmentation_data.TIMESTEP
position_list = []
direction_list = []
color_list = []
def build_model(env, num_procs, timesteps_per_actorbatch, optim_batchsize, output_dir):
    policy_kwargs = {
        "net_arch": [{"pi": [512, 256], "vf": [512, 256]}],
        "act_fun": tf.nn.relu,
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
        schedule="constant",
        policy_kwargs=policy_kwargs,
        tensorboard_log=output_dir,
        verbose=1,
    )
    return model


def test(model, env, num_episodes=None):
    o = env.reset()
    o *= 0
    data.set_rand_seed(1)
    trot = data.trot_array
    ma_v, _, _ = augmentation_data.calculate_ring_velocity(trot)
    position_list.append(trot)
    direction_list.append(ma_v)
    color_list.append(np.zeros((len(trot), 3)))
    point = augmentation_data.sample_random_point(trot)
    
    o[48:60] = point
    for _ in range(50):
        position_list.append(copy.deepcopy(point))
        # point_tensor = torch.tensor(point, dtype=torch.float32)
        a, _ = model.predict(o, deterministic=True)
        direction_list.append((a-point) / TIMESTEP)
    color_nums = len(position_list) - 1   
    color = np.ones((color_nums, 3))
    color[:, 0] = np.linspace(0.8, 0, color_nums)
    color_list.append(color)
    
    position_array = np.vstack(position_list)
    direction_array = np.vstack(direction_list)
    color_array = np.vstack(color_list)
    print(position_array.shape[0], direction_array.shape[0], color_array.shape[0])
    data.ploter(position_array, direction_array, color_array=color_array)

    env.close()


def main():

    num_procs = MPI.COMM_WORLD.Get_size()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    env = env_builder.build_imitation_env(
        motion_files=["motion_imitation/data/motions/dog_trot.txt"],
        num_parallel_envs=num_procs,
        mode="test",
        enable_randomizer=False,
        enable_rendering=False,
    )

    model = build_model(
        env=env,
        num_procs=num_procs,
        timesteps_per_actorbatch=TIMESTEPS_PER_ACTORBATCH,
        optim_batchsize=OPTIM_BATCHSIZE,
        output_dir="output",
    )

    model.load_parameters("motion_imitation/data/policies/dog_trot.zip")

    test(model=model, env=env, num_episodes=1)


if __name__ == "__main__":
    main()