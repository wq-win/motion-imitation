import os
import inspect
import tqdm
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
print(parentdir)
os.sys.path.insert(0, parentdir)
import copy
import numpy as np
import torch
from motion_imitation.envs import env_builder
from mpi4py import MPI
from pretrain import pretrain_save_data_V1


TIMESTEP = 1 / 30
DISPLACEMENT_RATE = 1

def oma_to_pma(oma):
    pma = copy.deepcopy(oma)
    pma[np.array([0, 6])] = -oma[np.array([0, 6])]
    pma[np.array([0, 3, 6, 9])] -= 0.30
    pma[np.array([1, 4, 7, 10])] -= 0.6
    pma[np.array([2, 5, 8, 11])] -= -0.66
    return pma


def oma_to_right_action(oma):
    action = copy.deepcopy(oma)
    # action[np.array([0, 3, 6, 9])] = -oma[np.array([0, 3, 6, 9])] #这里是错的
    action[np.array([1, 4, 7, 10])] -= 0.67
    action[np.array([2, 5, 8, 11])] -= -1.25
    return action


def main():
    test_model = pretrain_save_data_V1.Net(12, 12)
    test_model.load_state_dict(torch.load(os.path.join(parentdir,'pretrain_model/trot_data_model_07_04_21_11_10.pkl'), map_location=torch.device('cpu')))  

    env = env_builder.build_imitation_env(motion_files=[os.path.join(parentdir,"motion_imitation/data/motions/dog_trot.txt")],
                                        num_parallel_envs=MPI.COMM_WORLD.Get_size(),
                                        mode="test",
                                        enable_randomizer=False,
                                        enable_rendering=True,
                                        if_trajectory_generator=True)
    o = env.reset()
    error_factor = 0
    n_iter = 1  # 2
    ITER_FRACTER = 1/n_iter
    final_one = False
    action = np.zeros(12)
    max_kp = 600
    transition_kp=True
    for i in tqdm.tqdm(range(1500)):
        oma = o[48:60]
        pma = oma_to_pma(oma)
        pma = torch.tensor(pma, dtype=torch.float32)
        joint_velocity = test_model(pma)
        joint_velocity_array = np.zeros((n_iter, list(joint_velocity.shape)[0]))
        joint_velocity_array[0, :] = joint_velocity.detach().numpy()

        for i_iter in range(n_iter-1):
            joint_velocity = test_model(pma+joint_velocity*TIMESTEP*ITER_FRACTER)
            joint_velocity_array[i_iter+1, :] = joint_velocity.detach().numpy()
        
        joint_velocity_array[:, np.array([0, 6])] = -joint_velocity_array[:, np.array([0, 6])]
        
        if final_one:
            joint_velocity_average = joint_velocity_array[-1,:]
        else:
            joint_velocity_average = np.average(joint_velocity_array, axis=0)
        
        # joint_velocity_average = joint_velocity.detach().numpy()
        next_oma = oma + joint_velocity_average * TIMESTEP * DISPLACEMENT_RATE 
        action = oma_to_right_action(next_oma)

        motor_kps, motor_kds = env._gym_env._gym_env._gym_env.robot.GetMotorGains()
        if transition_kp and motor_kps[0]<max_kp:
            time_since_reset =  env._gym_env._gym_env._gym_env.robot.GetTimeSinceReset()
            motor_kps[:] = 220+(max_kp-220)*time_since_reset
        else:
            motor_kps[:] = max_kp
        # motor_kps = motor_kps
        env._gym_env._gym_env._gym_env.robot.SetMotorGains(motor_kps, motor_kds)
        o, r, d, _ = env.step(action)
        # if d:
        #     o = env.reset()

    env.close()

if __name__ == '__main__':
    main()