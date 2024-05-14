import argparse
from mpi4py import MPI
from motion_imitation.envs import env_builder as env_builder
import os
import numpy as np
from tqdm import tqdm
import pickle  
# python3 -m motion_imitation.examples.test_env_gui --robot_type=A1 --motor_control_mode=Position --on_rack=True
# python3 -m motion_imitation.examples.test_env_gui --robot_type=Laikago --motor_control_mode=Position --on_rack=True
# python3 -m motion_imitation.examples.test_env_gui --robot_type=A1 --motor_control_mode=Torque --on_rack=True
# python3 -m motion_imitation.examples.test_env_gui --robot_type=Laikago --motor_control_mode=Torque --on_rack=True
ENABLE_ENV_RANDOMIZER = True

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--seed", dest="seed", type=int, default=None)
arg_parser.add_argument("--mode", dest="mode", type=str, default="test")
arg_parser.add_argument("--motion_file", dest="motion_file", type=str, default="motion_imitation/data/motions/dog_pace.txt")
arg_parser.add_argument("--visualize", dest="visualize", action="store_true", default=True)
arg_parser.add_argument("--output_dir", dest="output_dir", type=str, default="output")
arg_parser.add_argument("--num_test_episodes", dest="num_test_episodes", type=int, default=None)
arg_parser.add_argument("--model_file", dest="model_file", type=str, default="")
arg_parser.add_argument("--total_timesteps", dest="total_timesteps", type=int, default=2e8)
arg_parser.add_argument("--int_save_freq", dest="int_save_freq", type=int, default=0) # save intermediate model every n policy steps

args = arg_parser.parse_args()

num_procs = MPI.COMM_WORLD.Get_size()
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

  
# 使用pickle从文件加载数组  
with open('weight_data.pkl', 'rb') as f:  
    allresult = pickle.load(f)  

PNtoKCweight = allresult['PNtoKCweight']
activate_KC_dims = allresult['activate_KC_dims']
KCtoMBONweight = allresult['KCtoMBONweight']

# PN dims > 2
# KC = PN @ PNtoKCweight
# sorted_indices = np.argsort(KC, axis=1)[:, ::-1]  
# inactivate_indices = sorted_indices[:, activate_KC_dims:]  
# KC[np.arange(KC.shape[0])[:, None], inactivate_indices] = 0
# MBON = KC @ KCtoMBONweight
# print(MBON)

enable_env_rand = ENABLE_ENV_RANDOMIZER and (args.mode != "test")
env = env_builder.build_imitation_env(motion_files=[args.motion_file],
                                        num_parallel_envs=num_procs,
                                        mode=args.mode,
                                        enable_randomizer=enable_env_rand,
                                        enable_rendering=args.visualize)

# print(env.observation_space, env.action_space)  # Box(160,) Box(12,)

o = env.reset()
env.render(mode='rgb_array')

# imu_quat=env._gym_env._gym_env._gym_env.robot.GetTrueBaseOrientation()
# imu_quat = [-imu_quat[0],-imu_quat[1],-imu_quat[2],imu_quat[3]]
# PN = np.append(imu_quat,o[48:60])

# IMU+MotorAngle
# PN = np.append(o[:4],o[48:60]) 
# goal
PN = o[87:103]

def cal_action(PN, PNtoKCweight, activate_KC_dims, KCtoMBONweight):
    KC = PN @ PNtoKCweight
    sorted_indices = np.argsort(KC)[::-1]  
    inactivate_indices = sorted_indices[activate_KC_dims:]  
    KC[inactivate_indices] = 0
    action = KC @ KCtoMBONweight
    return action

action = cal_action(PN, PNtoKCweight, activate_KC_dims, KCtoMBONweight)
# action *= 0
i = 0
# while i:
while True:
    # if i < 10:
    #     action *= 0.0002
    o, r, d, _ = env.step(action)
    # print(o[:12],o[12:24],o[48:60])
    # imu_quat = env._gym_env._gym_env._gym_env.robot.GetTrueBaseOrientation()
    # print(f'o:\n{o[84:84+19]}\nimu_quat:\n{imu_quat}')
    print(d)

    # imu_quat = [-imu_quat[0],-imu_quat[1],-imu_quat[2],imu_quat[3]]
    # PN = np.append(imu_quat,o[48:60])

    # print(imu_quat,np.linalg.norm(imu_quat))
    PN = o[87:103]
    # print(f'i:{i}\naction:\n{action}\no:\n{PN}')
    # PN *= 0.01

    action = cal_action(PN, PNtoKCweight, activate_KC_dims, KCtoMBONweight)
    action *= 0.1
    env.render(mode='rgb_array')
    if d:
        env.reset()
        
        i = 0
    print(action, i)    
    i += 1
env.close()
