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
    # pma[np.array([0, 3, 6, 9])] -= 0.15
    pma[np.array([0, 6])] = -oma[np.array([0, 6])]
    # pma[np.array([0, 3, 6, 9])] -= 0.3
    pma[np.array([1, 4, 7, 10])] -= 0.6
    # pma[np.array([2, 5, 8, 11])] -= -0.66
    pma[np.array([2, 5, 8, 11])] -= -0.75
    pma[np.array([2, 5, 8, 11])] -= -0.1
    return pma

def pma_to_oma(pma):
    oma = copy.deepcopy(pma)
    # pma[:, np.array([3, 9])] = -oma[:, np.array([3, 9])]

    oma[:, np.array([0, 6])] = -oma[:, np.array([0, 6])]
    # oma[:, np.array([0, 3, 6, 9])] += 0.15
    oma[:, np.array([1, 4, 7, 10])] += 0.6
    # oma[:, np.array([2, 5, 8, 11])] += -0.66
    oma[:, np.array([2, 5, 8, 11])] += -0.75
    return oma


def oma_to_right_action(oma):
    action = copy.deepcopy(oma)
    # action[np.array([0, 3, 6, 9])] = -oma[np.array([0, 3, 6, 9])] #这里是错的
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
    test_model.load_state_dict(torch.load(os.path.join(parentdir,'pretrain_model/trot_model_07_04_13_01.pkl'), map_location=torch.device('cpu')))

    env = env_builder.build_imitation_env(motion_files=[os.path.join(parentdir,"motion_imitation/data/motions/dog_trot.txt")],
                                        num_parallel_envs=MPI.COMM_WORLD.Get_size(),
                                        mode="test",
                                        enable_randomizer=False,
                                        enable_rendering=True,
                                        if_trajectory_generator=True)
    o = env.reset()
    error_factor = 0
    n_iter = 2
    ITER_FRACTER = 1/n_iter
    final_one = False
    transition_error = False
    action = np.zeros(12)
    max_kp = 300
    transition_kp=True
    for i in tqdm.tqdm(range(500)):
        # if transition_error and error_factor<1:
        #     error_factor = i*0.01
        # else:
        #     error_factor = 1
        tqdm.tqdm.write(str(error_factor))
        oma = o[48:60]
        print("action")
        print(action)
        print(oma)
        pma = oma_to_pma(oma)
        pma = torch.tensor(pma, dtype=torch.float32)
        joint_velocity = test_model(pma)
        joint_velocity_array = np.zeros((n_iter, list(joint_velocity.shape)[0]))
        joint_velocity_array[0, :] = joint_velocity.detach().numpy()

        for i_iter in range(n_iter-1):
            joint_velocity = test_model(pma+joint_velocity*TIMESTEP*ITER_FRACTER)
            joint_velocity_array[i_iter+1, :] = joint_velocity.detach().numpy()
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
        # action = oma_to_right_action(oma)
        # # action = [0.003*i for _ in range(12)]
        # action = np.zeros(12)
        # action[np.array([0,6])]=0.003*i
        # action[np.array([3, 9])] = -0.003 * i
        action_list.append(action)
        motor_kps, motor_kds = env._gym_env._gym_env._gym_env.robot.GetMotorGains()
        if transition_kp and motor_kps[0]<max_kp:
            time_since_reset =  env._gym_env._gym_env._gym_env.robot.GetTimeSinceReset()
            motor_kps[:] = 220+(max_kp-220)*time_since_reset
        else:
            motor_kps[:] = max_kp
        motor_kps = motor_kps
        env._gym_env._gym_env._gym_env.robot.SetMotorGains(motor_kps, motor_kds)
        o, r, d, _ = env.step(action)
        oma_list.append(o[48:60])
        # if d:
        #     o = env.reset()

    env.close()
    return test_model

def net_iter(net):
    input_list = []
    output_list = []
    for i in range(4):
        oma = np.random.uniform(-np.pi/2, np.pi/2, size=12)
        for _ in range(100):
            input_list.append(oma)
            oma = torch.tensor(oma, dtype=torch.float32)
            displacement = net(oma)
            displacement = displacement.detach().numpy()
            output_list.append(displacement)
            oma += displacement * TIMESTEP
    input_array = np.vstack(input_list)
    output_array  = np.vstack(output_list)
    return input_array, output_array

def trajectory_ploter(data, labels=[], axis=[0, 1, 2], ax=None):
    if ax is None:
        ax = plt.figure().add_subplot(projection='3d')

    for a_data, a_label in zip(data, labels):
        ax.plot(a_data[:, axis[0]], a_data[:, axis[1]], a_data[:, axis[2]], label=a_label)
    ax.legend()
    return ax

if __name__ == '__main__':
    test_model = main()
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
    with open(os.path.join(parentdir, 'collect_data/dataset/trot_data_V_07_04_10_100.pkl'), 'rb') as f:
    # with open(os.path.join(parentdir, 'collect_data/dataset/save_data_V5_model_06_21_11_06_43.pkl'), 'rb') as f:
        allresult = pickle.load(f)

    input = np.array(allresult['input'])
    output = np.array(allresult['output'])
    print(input.shape)
    input = pma_to_oma(input)
    output[:, np.array([0, 6])] = -output[:, np.array([0, 6])]
    from collect_data.save_data_V1_12D import quiver_ploter
    # for i in range(input.shape[0]//1000 // 20):
    #     trajactory_ploter(input, output, index_range=[i * 1000, (i + 1) * 1000], dim=num_joints, color_array=None,  x=0, y=1, z=2, u=0, v=1, w=2)


    input_array, output_array = net_iter(test_model)
    input_array = pma_to_oma(input_array)
    output_array[:, np.array([0, 6])] = -output_array[:, np.array([0, 6])]
    ax_list = []
    
    with open('test_model/ppo_track.pkl', 'rb') as f:
        allresult = pickle.load(f)
    ppo_position = np.array(allresult['input'])
    ppo_direction = np.array(allresult['output'])    
    
    # oma_list[:,np.array([0, 3, 6, 9])] += 0.02
    # oma_list[:,np.array([1, 4, 7, 10])] += 0.03
    # oma_list[:,np.array([2, 5, 8, 11])] += 0.1
    
    for i in range(4):
        start_index = 3*i
        ax = quiver_ploter(input, output, index_range=[0, 1000], dim=input.shape[1], color_array=None,
                           x=start_index, y=start_index+1, z=start_index+2,
                           u=start_index, v=start_index+1, w=start_index+2)
        ax_list.append(ax)
        ax_list[i] = trajectory_ploter([ oma_list, ppo_position],
                              labels=[ 'oma', 'ppo_track'],
                              axis=[start_index, start_index+1, start_index+2], ax=ax_list[i])
        ax_list[i] = quiver_ploter(input_array, output_array, index_range=[0, 1000], dim=input.shape[1], color_array=[0,0,0],
                                   x=start_index, y=start_index + 1, z=start_index + 2,
                                   u=start_index, v=start_index + 1, w=start_index + 2,
                                   ax=ax_list[i])


    plt.show()