import copy
import os
import inspect
import pickle

import numpy as np
import tqdm
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

from mpi4py import MPI
from motion_imitation.envs import env_builder


TIMESTEP = 1 / 30
episodes = 10
episode_length = 600

actions_list = []
episode_returns_list = []
rewards_list = []
obs_list = []
episode_starts_list = []

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
    action[:, np.array([0, 3, 6, 9])] = -oma[:, np.array([0, 3, 6, 9])]
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

    with open('dataset/augmentation_trot_data_10000_100.pkl', 'rb') as f:
        file_data = f.read()
        allresult = MPI.pickle.loads(file_data)
        print('data number', len(allresult['input']))
        assert len(allresult['input']) > episodes * episode_length, "not enough data"
        pma = allresult['input'][:episodes * episode_length,:]  # point
        velocity = allresult['output'][:episodes * episode_length,:]  # velocity
    
    oma = pma_to_oma(pma)    
    next_oma = oma + velocity * TIMESTEP
    action = oma_to_right_action(next_oma)
    o = env.reset()    
    for i in tqdm.tqdm(range(len(pma))):
        obs_list.append(o)
        actions_list.append(action[i])
        o, r, done, info = env.step(action[i])
        # print(r)
        rewards_list.append(r)
        episode_returns_list.append(r)
        episode_starts_list.append(done)
        if done: 
           o = env.reset()
           print('done' * 5)
        # print(f'{i} / {len(pma)}', end='\r')
    env.close()
    
if __name__ == '__main__':
    main()
    traj_data = {
        'episode_returns': np.array(episode_returns_list),
        'episode_starts': np.array(episode_starts_list),
        'obs': np.array(obs_list),
        'actions': np.array(actions_list),
        'rewards': np.array(rewards_list),
    }
    
    file_path = f'dataset/expertdata_{episodes}_{episode_length}.pkl'
    print(file_path)
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(file_path, 'wb') as f:
        pickle.dump(traj_data, f)