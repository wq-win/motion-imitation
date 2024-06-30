import os
import inspect

import tqdm



currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
print(parentdir)
os.sys.path.insert(0, parentdir)

import copy
import pickle
from matplotlib import pyplot as plt
import numpy as np
import torch
from motion_imitation.envs import env_builder
from mpi4py import MPI

from pretrain import pretrain_save_data_V1


TIMESTEP = 1 / 30
DISPLACEMENT_RATE = 1

action_list = []
without_error_action_list = []
oma_list = []
def oma_to_pma(oma):
    pma = copy.deepcopy(oma)
    # pma[np.array([3, 9])] = -oma[np.array([3, 9])]
    pma[np.array([0, 6])] = -oma[np.array([0, 6])]
    pma[np.array([1, 4, 7, 10])] -= 0.6
    pma[np.array([2, 5, 8, 11])] -= -0.66
    return pma


def oma_to_right_action(oma):
    action = copy.deepcopy(oma)
    action[np.array([0, 3, 6, 9])] = -oma[np.array([0, 3, 6, 9])]
    action[np.array([1, 4, 7, 10])] -= 0.67
    action[np.array([2, 5, 8, 11])] -= -1.25
    return action


def error_between_target_and_result(o, ignore_hip=False):
    """
    target motorangle is o[12:24]=env.step input action
    result motorangle is o[48:60]=current observation motorangle
    """
    error = o[12:24] - o[48:60]
    if ignore_hip:
        error[np.array([0, 3, 6, 9])] = 0
    return error


def a_to_oa(a):
    oa = copy.deepcopy(a)
    oa[:, np.array([1, 4, 7, 10])] += 0.67
    oa[:, np.array([2, 5, 8, 11])] += -1.25
    return oa

def main():
    test_model = pretrain_save_data_V1.Net(12, 12)
    test_model.load_state_dict(torch.load(os.path.join(parentdir,'pretrain_model/save_data_V5_model_06_21_11_06_43.pkl'), map_location=torch.device('cpu')))

    env = env_builder.build_imitation_env(motion_files=[os.path.join(parentdir,"motion_imitation/data/motions/dog_pace.txt")],
                                        num_parallel_envs=MPI.COMM_WORLD.Get_size(),
                                        mode="test",
                                        enable_randomizer=False,
                                        enable_rendering=True,
                                        if_trajectory_generator=True)
    o = env.reset()
    error_factor = 0
    n_iter = 2
    final_one = True
    transition_error = False
    for i in tqdm.tqdm(range(500)):
        if transition_error and error_factor<1:
            error_factor = i*0.01
        else:
            error_factor = 1
        tqdm.tqdm.write(str(error_factor))
        oma = o[48:60]
        pma = oma_to_pma(oma)
        pma = torch.tensor(pma, dtype=torch.float32)
        joint_velocity = test_model(pma)
        joint_velocity_array = np.zeros((n_iter, list(joint_velocity.shape)[0]))
        joint_velocity_array[0, :] = joint_velocity.detach().numpy()

        for i in range(n_iter-1):
            joint_velocity = test_model(pma+joint_velocity*TIMESTEP)
            joint_velocity_array[i+1, :] = joint_velocity.detach().numpy()
        joint_velocity_array[:, np.array([0, 6])] = -joint_velocity_array[:, np.array([0, 6])]
        # joint_velocity = joint_velocity.detach().numpy()
        # joint_velocity[np.array([0, 6])] = -joint_velocity[np.array([0, 6])]
        # joint_velocity[np.array([3, 9])] = -joint_velocity[np.array([3, 9])]
        if final_one:
            joint_velocity_average = joint_velocity_array[-1,:]
        else:
            joint_velocity_average = np.average(joint_velocity_array, axis=0)
        without_error_oma = oma + joint_velocity_average * TIMESTEP * DISPLACEMENT_RATE
        without_error_oma_action = oma_to_right_action(without_error_oma)
        without_error_action_list.append(without_error_oma_action)

        next_oma = oma + joint_velocity_average * TIMESTEP * DISPLACEMENT_RATE + error_between_target_and_result(o, True) * error_factor
        action = oma_to_right_action(next_oma)
        action_list.append(action)
        o, r, d, _ = env.step(action)
        oma_list.append(o[48:60])
        # if d:
        #     o = env.reset()

    env.close()
    
if __name__ == '__main__':
    main()
    action_list = np.array(action_list)
    without_error_action_list = np.array(without_error_action_list)
    oma_list = np.array(oma_list)

    action_list = a_to_oa(action_list)
    without_error_action_list = a_to_oa(without_error_action_list)
    plt.figure()
    for i in range(12):
        plt.subplot(4, 3, i+1)
        plt.plot(range(len(action_list[:, i]),), action_list[:, i], label=f'action:{i}', linestyle='-')
        plt.plot(range(len(without_error_action_list[:, i]),), without_error_action_list[:, i], label=f'without_error_action:{i}', linestyle='-.')
        plt.plot(range(len(oma_list[:, i]),), oma_list[:, i], label=f'oma:{i}', linestyle='--')
        plt.legend()
    plt.show()

    # with open('dataset/save_data_V4_100000_10.pkl', 'rb') as f:
    with open(os.path.join(parentdir, 'collect_data/dataset/save_data_V06_29_10_100.pkl'), 'rb') as f:
    # with open(os.path.join(parentdir, 'collect_data/dataset/save_data_V5_model_06_21_11_06_43.pkl'), 'rb') as f:
        allresult = pickle.load(f)

    input = np.array(allresult['input'])
    output = np.array(allresult['output'])
    print(input.shape)
    from collect_data.save_data_V1_12D import trajactory_ploter
    # for i in range(input.shape[0]//1000 // 20):
    #     trajactory_ploter(input, output, index_range=[i * 1000, (i + 1) * 1000], dim=num_joints, color_array=None,  x=0, y=1, z=2, u=0, v=1, w=2)
    ax = trajactory_ploter(input, output, index_range=[0, 1000], dim=input.shape[1], color_array=None, x=0, y=1, z=2, u=0,
                           v=1, w=2)

    # ax = plt.figure().add_subplot(projection='3d')

    x1= action_list[:, 0]
    y1= action_list[:, 1]
    z1= action_list[:, 2]
    ax.plot(x1, y1, z1, label='action')
    x2= without_error_action_list[:, 0]
    y2= without_error_action_list[:, 1]
    z2= without_error_action_list[:, 2]
    ax.plot(x2, y2, z2, label='without_error_action')
    x3= oma_list[:, 0]
    y3= oma_list[:, 1]
    z3= oma_list[:, 2]
    ax.plot(x3, y3, z3, label='oma')
    ax.legend()

    plt.show()