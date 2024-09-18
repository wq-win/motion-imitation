import copy
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
NOWTIME = time.strftime("%m_%d_%H_%M_%S", time.localtime())

def pma_to_oma(pma):
    oma = copy.deepcopy(pma)
    # oma[np.array([0, 6])] = -pma[np.array([0, 6])]
    oma[np.array([0, 6])] -= 0.30
    oma[np.array([3, 9])] += 0.30
    oma[np.array([1, 4, 7, 10])] += 0.6
    oma[np.array([2, 5, 8, 11])] += -0.66
    return oma

def oma_to_right_action(oma):
    action = copy.deepcopy(oma)
    action[np.array([1, 4, 7, 10])] -= 0.67
    action[np.array([2, 5, 8, 11])] -= -1.25
    return action

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
    # episode_length = 600
    # collect_episodes = 1
    # i = 0
    while True:
        # o[:48] = 0
        # o[60:] = 0
        # print(f'i:{i}/{episode_length * collect_episodes}', end="\r")
        a, _ = model.predict(o, deterministic=True)
        # a = pma_to_oma(a)
        # a = oma_to_right_action(a)
        o, r, done, info = env.step(a)
        print(r)
        if done:
            print(f'done:{done}')
            o = env.reset()
        # if i >= episode_length * collect_episodes:
        #     break  
        # i += 1  
    env.close()


def main():

    num_procs = MPI.COMM_WORLD.Get_size()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    env = env_builder.build_imitation_env(
        motion_files=["motion_imitation/data/motions/dog_trot.txt"],
        num_parallel_envs=num_procs,
        mode="test",
        enable_randomizer=False,
        enable_rendering=True,
    )

    model = build_model(
        env=env,
        num_procs=num_procs,
        timesteps_per_actorbatch=TIMESTEPS_PER_ACTORBATCH,
        optim_batchsize=OPTIM_BATCHSIZE,
        output_dir="output",
    )

    # model.load_parameters('E:\VScode\motion-imitation\output\pretrained_model\pretrained_model_100_steps_2024_07_24_13_17_27.zip')
    # model.load_parameters('E:\VScode\motion-imitation\output\pretrained_model\pretrained_model_100_steps_2024_08_09_12_18_52.zip')
    # model.load_parameters('E:\VScode\motion-imitation\output\pretrained_model\pretrained_model_100_steps_2024_08_09_16_05_39.zip')

    # model.load_parameters('E:\VScode\motion-imitation\output\model_8_01_no_3trick_4e7_no_load.zip')
    # model.load_parameters('E:\VScode\motion-imitation\output\model_8_02_no_3trick_4e7_load_pretrain.zip')
    model.load_parameters('E:\VScode\motion-imitation\motion_imitation\data\policies\dog_trot.zip')
    # model.load_parameters('E:\VScode\motion-imitation\output\model_8_04_no_load_2e8.zip')
    # model.load_parameters('E:\VScode\motion-imitation\output\model_8_11_load_2e8.zip')
    
    model.load_parameters('E:\VScode\motion-imitation\output\expertdata_model\expertdata_10.zip')
    # model.load_parameters('E:\VScode\motion-imitation\output\expertdata_model\expertdata_100.zip')
    # model.load_parameters('E:\VScode\motion-imitation\output\expertdata_model\expertdata_1000.zip')

    test(model=model, env=env, num_episodes=1000)



if __name__ == "__main__":
    main()
