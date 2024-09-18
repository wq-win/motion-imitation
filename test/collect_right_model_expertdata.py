import copy
import os
import inspect
import pickle

import numpy as np
import tqdm
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

from mpi4py import MPI
from motion_imitation.envs import env_builder
from motion_imitation.learning import ppo_imitation as ppo_imitation
from motion_imitation.learning import imitation_policies as imitation_policies
from motion_imitation.envs import env_builder as env_builder
import tensorflow as tf
import numpy as np
from mpi4py import MPI


TIMESTEPS_PER_ACTORBATCH = 4096
OPTIM_BATCHSIZE = 256

TIMESTEP = 1 / 30
episodes = 10
episode_length = 600

actions_list = []
episode_returns_list = []
rewards_list = []
obs_list = []
episode_starts_list = []


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


def main():
    num_procs = MPI.COMM_WORLD.Get_size()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    env = env_builder.build_imitation_env(motion_files=[os.path.join(parentdir,"motion_imitation/data/motions/dog_trot.txt")],
                                    num_parallel_envs=MPI.COMM_WORLD.Get_size(),
                                    mode="test",
                                    enable_randomizer=False,
                                    enable_rendering=False,
                                    if_trajectory_generator=True)

    
    model = build_model(
        env=env,
        num_procs=num_procs,
        timesteps_per_actorbatch=TIMESTEPS_PER_ACTORBATCH,
        optim_batchsize=OPTIM_BATCHSIZE,
        output_dir="output",
    )
    model.load_parameters('E:\VScode\motion-imitation\motion_imitation\data\policies\dog_trot.zip')

    o = env.reset()    
    for i in tqdm.tqdm(range(episodes * episode_length)):
        a, _ = model.predict(o, deterministic=True)
        obs_list.append(o)
        actions_list.append(a)
        o, r, done, info = env.step(a)
        # print(r)
        rewards_list.append(r)
        episode_returns_list.append(r)
        episode_starts_list.append(done)
        if done: 
           o = env.reset()
        # print(f'{i} / {len(pma)}', end='\r')
    env.close()
    
if __name__ == '__main__':
    main()
    traj_data = {
        'episode_returns': np.array(episode_returns_list),
        'episode_starts': np.array(episode_starts_list),
        'obs': np.array(obs_list),
        'actions': np.array(actions_list),
        'rewards': np.array(rewards_list),
    }
    
    file_path = f'dataset/expertdata_right_{episodes}_{episode_length}.pkl'
    print(file_path)
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(file_path, 'wb') as f:
        pickle.dump(traj_data, f)