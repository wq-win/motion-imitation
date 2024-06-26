import inspect
import os
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

from matplotlib import pyplot as plt
import numpy as np
from motion_imitation.envs import env_builder as env_builder
from mpi4py import MPI
import tensorflow as tf
from motion_imitation.learning import imitation_policies as imitation_policies
from motion_imitation.learning import ppo_imitation as ppo_imitation



a_list = []
oma_list = []

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


def test(model, env):
    RUNTIMES = 600
    o = env.reset()
    while True:
        a, _ = model.predict(o, deterministic=True)
        o, r, done, info = env.step(a)
        # print(f'a:\n{a}')
        # print(f'oma:\n{o[48:60]}')
        a_list.append(a)
        oma_list.append(o[48:60])
        if done:
            o = env.reset()
        if RUNTIMES == 0:
             break
        RUNTIMES -= 1
    env.close()


env = env_builder.build_imitation_env(motion_files=["motion_imitation/data/motions/dog_pace.txt"],
                                    num_parallel_envs=MPI.COMM_WORLD.Get_size(),
                                    mode="test",
                                    enable_randomizer=False,
                                    enable_rendering=False)
model = build_model(env=env,
                    num_procs=MPI.COMM_WORLD.Get_size(),
                    timesteps_per_actorbatch=4096,
                    optim_batchsize=256,
                    output_dir='output')

model.load_parameters("motion_imitation/data/policies/dog_pace.zip")
test(model=model, env=env)
oma_list = np.array(oma_list)
a_list = np.array(a_list)
a_list[:, np.array([1, 4, 7, 10])] += 0.67
a_list[:, np.array([2, 5, 8, 11])] += -1.25
for i in range(12):
        plt.subplot(4, 3, i+1)
        plt.plot(range(len(oma_list[102:402, i]),), oma_list[102:402, i], label=f'oma:{i}', linestyle='-')
        plt.plot(range(len(a_list[100:400, i])), a_list[100:400, i], label=f'a:{i}', linestyle='--')
        plt.legend()
        
file_path = f"result/compare/a_oma.png"
directory = os.path.dirname(file_path)
if not os.path.exists(directory):
    os.makedirs(directory)
plt.savefig(file_path, dpi=300)    
plt.show()  
