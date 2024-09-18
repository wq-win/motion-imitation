import copy
from mpi4py import MPI
import numpy as np
from stable_baselines.gail.dataset.dataset import ExpertDataset


TIMESTEP = 1 / 30


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


with open('dataset/augmentation_trot_data_100000_100.pkl', 'rb') as f:
    file_data = f.read()
    
    allresult = MPI.pickle.loads(file_data)
    preobs = allresult['input'][::10,:]  # point
    actions = allresult['output'][::10,:]  # velocity
    
    pma = preobs
    velocity = actions
    oma = pma_to_oma(pma)
    next_oma = oma + velocity * TIMESTEP
    action = oma_to_right_action(next_oma)
    
    del allresult
    obs = np.zeros((len(preobs),160))
    
    obs[:, 48:60] = oma
    # create the dataset
    
    traj_data = {'obs': obs,
                'actions': action,
                'episode_returns': np.zeros(len(preobs), dtype=bool),
                'rewards': np.zeros(len(preobs), dtype=bool),
                'episode_starts': np.zeros(len(preobs), dtype=bool)}
    # 'actions', 'episode_returns', 'rewards', 'obs' and 'episode_starts'
    dataset = ExpertDataset(traj_data=traj_data, )
    print(dataset.observations.shape)
    print(dataset.actions.shape)
    print(dataset.returns.shape)
    print(dataset.num_traj)
