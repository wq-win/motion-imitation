from matplotlib import pyplot as plt
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
ppo_ma_track_list = []
action_list = []
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
    episode_length = 120
    if num_episodes is None:
        num_episodes = 1
    else:
        num_episodes = num_episodes
    i = 0
    while True:
        print(f'i:{i}/{episode_length * num_episodes}', end="\r")
        ppo_ma_track_list.append(o[48:60])
        a, _ = model.predict(o, deterministic=True)
        action_list.append(a)
        o, r, done, info = env.step(a)
        if done:
            o = env.reset()
        if i >= episode_length * num_episodes:
            break  
        i += 1  
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
    ppo_ma_track_array = np.array(ppo_ma_track_list)
    ppo_action_array = all_offset.oma_to_right_action(ppo_ma_track_array)
    right_action_v, _, _ = augmentation_data.calculate_ring_velocity(ppo_action_array)
    # position_list.append(ppo_action_array)
    # direction_list.append(right_action_v)
    # color_nums_p = len(ppo_action_array) - 1   
    # color_p = np.ones((color_nums_p, 3))
    # color_p[:, 0] = np.linspace(0.8, 0, color_nums_p)
    # color_list.append(color_p) 

    
    action_array = np.array(action_list)
    action_v, _, _ = augmentation_data.calculate_ring_velocity(action_array)
    # position_list.append(action_array)
    # direction_list.append(action_v)
    # color_nums = len(action_array) - 1   
    # color = np.ones((color_nums, 3))
    # color[:, 1] = np.linspace(0.8, 0, color_nums)
    # color_list.append(color) 
    
    # position_array = np.vstack(position_list)
    # direction_array = np.vstack(direction_list)
    # color_array = np.vstack(color_list)
    # print(position_array.shape[0], direction_array.shape[0], color_array.shape[0])
    # data.ploter(position_array, direction_array, color_array=color_array)
    DISCARD_INDEX = 10
    OFFSET_INDEX = 3
    ppo_action_array = ppo_action_array[DISCARD_INDEX + OFFSET_INDEX: , :]
    action_array = action_array[DISCARD_INDEX : , :]
    plt.figure()
    for i in range(12):
        plt.subplot(4, 3, i+1)
        plt.plot(range(len(ppo_action_array[:, i]),), ppo_action_array[:, i], label=f'ppo_ma:{i}', linestyle='-')
        plt.plot(range(len(action_array[:, i]),), action_array[:, i], label=f'action:{i}', linestyle='--')
        plt.legend()
    plt.show()