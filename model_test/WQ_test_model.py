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

from pretrain import pretrain_oma_data_Net
from collect import collect_pma_data
from collect_test import test_oma
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


def ploter(position1, direction1, position2, direction2, x=0, y=1, z=2, u=0, v=1, w=2):
    ax = plt.figure().add_subplot(projection='3d')
    X1 = position1[:, x]
    Y1 = position1[:, y]
    Z1 = position1[:, z]
    U1 = direction1[:, u]
    V1 = direction1[:, v]
    W1 = direction1[:, w]
    color_array1 = np.ones((position1.shape[0],3))
    color_array1[:, 1] = np.linspace(0.8, 0, color_array1.shape[0])
    ax.quiver(X1, Y1, Z1, U1, V1, W1, color=color_array1, length=TIMESTEP, normalize=False, label='oma')
    
    X2 = position2[:, x]
    Y2 = position2[:, y]
    Z2 = position2[:, z]
    U2 = direction2[:, u]
    V2 = direction2[:, v]
    W2 = direction2[:, w]
    color_array2 = np.ones((position2.shape[0],3))
    color_array2[:, 0] = np.linspace(0.8, 0, color_array2.shape[0])
    ax.quiver(X2, Y2, Z2, U2, V2, W2, color=color_array2, length=TIMESTEP, normalize=False, label='ppo')
    
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.legend()
    ax.set_title('our model with ppo')
    test_oma.set_axes_equal(ax)
    plt.show()

def main():
    test_model = pretrain_oma_data_Net.Net(12, 12)
    test_model.load_state_dict(torch.load(os.path.join(parentdir,'pretrain_model/oma_model_06_26_16_51.pkl'), map_location=torch.device('cpu')))

    env = env_builder.build_imitation_env(motion_files=[os.path.join(parentdir,"motion_imitation/data/motions/dog_pace.txt")],
                                        num_parallel_envs=MPI.COMM_WORLD.Get_size(),
                                        mode="test",
                                        enable_randomizer=False,
                                        enable_rendering=True,
                                      )
    o = env.reset()
    error_factor = 0
    n_iter = 2
    ITER_FRACTER = 1/n_iter
    final_one = False
    transition_error = False
    action = np.zeros(12)
    max_kp = 300
    transition_kp=True
    for i in tqdm.tqdm(range(600)):
        tqdm.tqdm.write(str(error_factor))
        oma = o[48:60]
        # print("action")
        # print(action)
        # print(oma)
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
        without_error_oma = oma + joint_velocity_average * TIMESTEP * DISPLACEMENT_RATE
        without_error_oma_action = oma_to_right_action(without_error_oma)
        without_error_action_list.append(without_error_oma_action)

        next_oma = oma + joint_velocity_average * TIMESTEP * DISPLACEMENT_RATE + error_between_target_and_result(o, True) * error_factor
        action = oma_to_right_action(next_oma)

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
    oma_array = np.array(oma_list)
    oma_v, _, _ = collect_pma_data.calculate_ring_velocity(oma_array)
    action_list = a_to_oa(action_list)
    without_error_action_list = a_to_oa(without_error_action_list)
    with open('function_test/ppo_track.pkl', 'rb') as f:
        allresult = pickle.load(f)
    p2 = np.array(allresult['input'])
    d2 = np.array(allresult['output'])    
    ploter(oma_array, oma_v, p2, d2)
