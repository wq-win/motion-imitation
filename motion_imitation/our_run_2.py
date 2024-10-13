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

import logging
logging.getLogger().setLevel(logging.ERROR)


NOWTIME = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
NOWTIME_M_D = time.strftime("%m_%d", time.localtime())
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


def train(model, env, total_timesteps, output_dir="", int_save_freq=0, pretrain_flag=False, output_is_speed=False, load_model_flag=False,):
    if (output_dir == ""):
        save_path = None
    else:
        save_path = os.path.join(output_dir, f"model_{NOWTIME_M_D}_final_our_model_{total_timesteps}_{int_save_freq}.zip")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    callbacks = []
    # Save a checkpoint every n steps
    if (output_dir != ""):
        if (int_save_freq > 0):
            int_dir = os.path.join(output_dir, "intermedate")
            callbacks.append(CheckpointCallback(save_freq=int_save_freq, save_path=int_dir,
                                                name_prefix=f'model_{NOWTIME}'))
        
    # if pretrain_flag:
    #     save_file_path=f'output/expertdata_model/expertdata_10.zip'
    #     with open('dataset/expertdata_10_600.pkl', 'rb') as f:
    #         file_data = f.read()
    #         traj_data = MPI.pickle.loads(file_data)
    # print(model.policy_pi.trainable_variables)
    print(f'MPI.COMM_WORLD.Get_rank() is {MPI.COMM_WORLD.Get_rank()} , size is {MPI.COMM_WORLD.Get_size()}')
    if MPI.COMM_WORLD.Get_rank() == 0:
        # with open('E:\VScode\motion-imitation\dataset\expertdata_10_600.pkl', 'rb') as f:
        with open('/home/user2020/VScodeProjects/motion-imitation/dataset/expertdata_10_600.pkl', 'rb') as f:

            file_data = f.read()
            traj_data = MPI.pickle.loads(file_data)
        # dataset = ExpertDataset(traj_data=traj_data, )
        # model.pretrain(dataset, n_epochs=100, learning_rate=1e-4, adam_epsilon=1e-8)  
    else:
        traj_data = None

    # 广播读取的 traj_data 给其他进程
    traj_data = MPI.COMM_WORLD.bcast(traj_data, root=0)
       
 
    dataset = ExpertDataset(traj_data=traj_data, )
    model.pretrain(dataset, n_epochs=100, learning_rate=1e-4, adam_epsilon=1e-8,)
    #     directory = os.path.dirname(save_file_path)
    #     if not os.path.exists(directory):
    #         os.makedirs(directory)
    #     model.save(save_file_path)
    
    # return
    if load_model_flag:
        load_file_path = f'output/expertdata_model/expertdata_10.zip'
        model.load_parameters(load_file_path)
    # model.adam.sync()
    model.learn(total_timesteps=total_timesteps, save_path=save_path,
                callback=callbacks, output_is_speed=output_is_speed)

    return


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--seed", dest="seed", type=int, default=0)
    arg_parser.add_argument("--mode", dest="mode", type=str, default="train")
    arg_parser.add_argument("--motion_file", dest="motion_file",
                            type=str, default="motion_imitation/data/motions/dog_trot.txt")
    arg_parser.add_argument("--visualize", dest="visualize",
                            action="store_true", default=False)
    arg_parser.add_argument(
        "--output_dir", dest="output_dir", type=str, default="output")
    arg_parser.add_argument("--num_test_episodes",
                            dest="num_test_episodes", type=int, default=None)
    arg_parser.add_argument(
        "--model_file", dest="model_file", type=str, default="")
    arg_parser.add_argument("--total_timesteps",
                            dest="total_timesteps", type=int, default=2e8)
    # save intermediate model every n policy steps
    arg_parser.add_argument(
        "--int_save_freq", dest="int_save_freq", type=int, default=1e6)
    arg_parser.add_argument(
        "--pretrain_flag", dest="pretrain_flag", type=bool, default=False)
    arg_parser.add_argument("--output_is_speed",
                            dest="output_is_speed", type=bool, default=False)
    arg_parser.add_argument("--load_model_flag",
                            dest="load_model_flag", type=bool, default=False)

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
    print('pretrain_flag:', args.pretrain_flag)
    print('load_model_flag:', args.load_model_flag)
    if args.mode == "train":
        train(model=model,
              env=env,
              total_timesteps=args.total_timesteps,
              output_dir=args.output_dir,
              int_save_freq=args.int_save_freq,
              pretrain_flag=args.pretrain_flag,
              output_is_speed=args.output_is_speed,
              load_model_flag=args.load_model_flag,)
    else:
        assert False, "Unsupported mode: " + args.mode

    return


if __name__ == '__main__':
    main()
