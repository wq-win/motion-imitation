import matplotlib.pyplot as plt
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
import os
import inspect
import pickle
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

# import tf_slim as slim


TIMESTEPS_PER_ACTORBATCH = 4096
OPTIM_BATCHSIZE = 256
NOWTIME = time.strftime("%m-%d_%H-%M-%S", time.localtime())
# start_time = time.time()
ENABLE_ENV_RANDOMIZER = True
l = list()


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


current_timestamp = time.time()
local_time = time.localtime(current_timestamp)
formatted_time = time.strftime("-%m-%d-", local_time)


def test(model, env, num_procs, num_episodes=None):
    i = 0
    EPISODE = 600  # one episode has 600 step
    collect_nums = 10000
    o_list = []
    a_list = []
    o = env.reset()
    while i < EPISODE * collect_nums:
        i += 1
        a, _ = model.predict(o, deterministic=True)
        o_list.append(o)
        a_list.append(a)
        o, r, done, info = env.step(a)

        if done:
            o = env.reset()
            print(f"eposide:{i//600}/{collect_nums}")
            # print(time.time()- start_time)
    env.close()
    allresult = {'o': o_list, 'a': a_list}
    with open('pretrain/dataset/oa_{}.pkl'.format(NOWTIME), 'wb') as f:
        pickle.dump(allresult, f)
    return


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--seed", dest="seed", type=int, default=None)
    arg_parser.add_argument("--mode", dest="mode", type=str, default="test")
    arg_parser.add_argument("--motion_file", dest="motion_file",
                            type=str, default="motion_imitation/data/motions/dog_pace.txt")
    arg_parser.add_argument("--visualize", dest="visualize",
                            action="store_true", default=False)
    arg_parser.add_argument(
        "--output_dir", dest="output_dir", type=str, default="output")
    arg_parser.add_argument("--num_test_episodes",
                            dest="num_test_episodes", type=int, default=None)
    arg_parser.add_argument(
        "--model_file", dest="model_file", type=str, default="")
    arg_parser.add_argument("--total_timesteps",
                            dest="total_timesteps", type=int, default=2e5)
    # save intermediate model every n policy steps
    arg_parser.add_argument(
        "--int_save_freq", dest="int_save_freq", type=int, default=0)

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
             num_procs=num_procs,
             num_episodes=args.num_test_episodes)
    else:
        assert False, "Unsupported mode: " + args.mode
    return


if __name__ == '__main__':
    main()
