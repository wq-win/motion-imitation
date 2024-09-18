import copy
import os
import inspect

import numpy as np
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

from mpi4py import MPI
from motion_imitation.envs import env_builder


TIMESTEP = 1 / 30
max_kp = 600

# def pma_to_oma(pma):
#     oma = copy.deepcopy(pma)
#     oma[:, np.array([0, 6])] = -pma[:, np.array([0, 6])]
#     oma[:, np.array([0, 6])] += 0.3
#     oma[:, np.array([3, 9])] -= 0.3
#     oma[:, np.array([1, 4, 7, 10])] += 0.6
#     oma[:, np.array([2, 5, 8, 11])] += -0.66
#     return oma
# def oma_to_right_action(oma):
#     action = copy.deepcopy(oma)
#     action[:, np.array([0, 3, 6, 9])] = -oma[:, np.array([0, 3, 6, 9])]
#     action[:, np.array([1, 4, 7, 10])] -= 0.67
#     action[:, np.array([2, 5, 8, 11])] -= -1.25
#     return action

def pma_to_oma(pma):
    oma = copy.deepcopy(pma)
    # oma[:, np.array([0, 6])] = -pma[:, np.array([0, 6])]
    oma[:, np.array([3, 9])] = -pma[:, np.array([3, 9])]
    oma[:, np.array([0, 6])] -= 0.3
    oma[:, np.array([3, 9])] += 0.3
    oma[:, np.array([1, 4, 7, 10])] += 0.6
    oma[:, np.array([2, 5, 8, 11])] += -0.66
    return oma
def oma_to_right_action(oma):
    action = copy.deepcopy(oma)
    # action[:, np.array([0, 6])] = -oma[:, np.array([0, 6])]
    # action[:, np.array([0, 3, 6, 9])] = -oma[:, np.array([0, 3, 6, 9])]
    action[:, np.array([1, 4, 7, 10])] -= 0.67
    action[:, np.array([2, 5, 8, 11])] -= -1.25
    return action

def main():
    env = env_builder.build_imitation_env(motion_files=[os.path.join(parentdir,"motion_imitation/data/motions/dog_trot.txt")],
                                    num_parallel_envs=MPI.COMM_WORLD.Get_size(),
                                    mode="test",
                                    enable_randomizer=False,
                                    enable_rendering=True,
                                    if_trajectory_generator=True)

    with open('dataset/augmentation_trot_data_100000_100.pkl', 'rb') as f:
        file_data = f.read()
        
        allresult = MPI.pickle.loads(file_data)
        pma = allresult['input'][:1000,:]  # point
        velocity = allresult['output'][:1000,:]  # velocity
    
    oma = pma_to_oma(pma)    
    next_oma = oma + velocity * TIMESTEP
    action = oma_to_right_action(next_oma)
    # motor_kps, motor_kds = env._gym_env._gym_env._gym_env.robot.GetMotorGains()
    # if motor_kps[0]<max_kp:
    #     time_since_reset =  env._gym_env._gym_env._gym_env.robot.GetTimeSinceReset()
    #     motor_kps[:] = 220+(max_kp-220)*time_since_reset
    # else:
    #     motor_kps[:] = max_kp
    # env._gym_env._gym_env._gym_env.robot.SetMotorGains(motor_kps, motor_kds)
        
    for i in range(len(pma)):
        o, r, done, _ = env.step(action[i])
        # if done: 
        #     print('done', end='\n')
        #     env.reset()
        # print(f'{i} / {len(pma)}', end='\r')
    env.close()
    
if __name__ == '__main__':
    main()