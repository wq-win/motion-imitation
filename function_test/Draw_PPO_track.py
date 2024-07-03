import os
import inspect
import pickle
import time
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

from stable_baselines.common.callbacks import CheckpointCallback
from motion_imitation.learning import ppo_imitation as ppo_imitation
from motion_imitation.learning import imitation_policies as imitation_policies
from motion_imitation.envs import env_builder as env_builder
import tensorflow as tf
import numpy as np
from mpi4py import MPI
from collect import collect_pma_data
from collect_test import test_pma

TIMESTEPS_PER_ACTORBATCH = 4096
OPTIM_BATCHSIZE = 256
ppo_ma_track_list = []
NOWTIME = time.strftime("%m_%d", time.localtime())

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
    episode_length = 600
    collect_episodes = 1
    i = 0
    point = collect_pma_data.sample_random_point_pi()
    o[48:60] = point
    while True:
        print(f'i:{i}/{episode_length * collect_episodes}', end="\r")
               
        ppo_ma_track_list.append(o[48:60])
        a, _ = model.predict(o, deterministic=True)
        o, r, done, info = env.step(a)
        if done:
            o = env.reset()
        if i >= episode_length * collect_episodes:
            break  
        i += 1  
    env.close()


def main():

    num_procs = MPI.COMM_WORLD.Get_size()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    env = env_builder.build_imitation_env(
        motion_files=["motion_imitation/data/motions/dog_pace.txt"],
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

    model.load_parameters("motion_imitation/data/policies/dog_pace.zip")

    test(model=model, env=env, num_episodes=1000)


if __name__ == "__main__":
    main()
    ppo_ma_track_array = np.array(ppo_ma_track_list)
    ma_v, ma_v_norm, ma_weight = collect_pma_data.calculate_ring_velocity(ppo_ma_track_array)
    test_pma.ploter(ppo_ma_track_array, ma_v)
    allresult = {'input': ppo_ma_track_array, 'output': ma_v, }
    file_path = f'function_test/ppo_track_{NOWTIME}.pkl'
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(file_path, 'wb') as f:
        pickle.dump(allresult, f)