import copy
import pickle
from matplotlib import pyplot as plt
import numpy as np
import data_augmentation


trot = data_augmentation.tort
JOINT_INDEX_START = data_augmentation.JOINT_INDEX_START
JOINT_NUMS = data_augmentation.JOINT_NUMS
trot_array = np.array(trot)[:, JOINT_INDEX_START: JOINT_INDEX_START + JOINT_NUMS]


def trot_to_oma(trot: np.ndarray)->np.ndarray:
    oma = copy.deepcopy(trot)
    if len(oma.shape) > 1 and oma.shape[0] > 1:
        oma[:, np.array([0, 6])] = -trot[:, np.array([0, 6])]
        oma[:, np.array([1, 4, 7, 10])] -= -0.6
        oma[:, np.array([2, 5, 8, 11])] -= 0.66
    else:
        oma[np.array([0, 6])] = -trot[np.array([0, 6])]
        oma[np.array([1, 4, 7, 10])] -= -0.6
        oma[np.array([2, 5, 8, 11])] -= 0.66
    return oma

if __name__ == '__main__':
    with open('trot/dataset/collect_a_oma_data.pkl', 'rb') as f:
        allresult = pickle.load(f)
        
    o = allresult['o']
    o = np.array(o)
    oma = o[205:305, 48:60]
    trot_array = trot_to_oma(trot_array)
    trot_array = np.tile(trot_array, (7, 1))
    trot_array = trot_array[23:, :]
    plt.figure()
    for i in range(12):
        plt.subplot(4, 3, i+1)
        plt.plot(range(0, 2*len(oma[:, i]),2), oma[:, i], label=f'oma:{i}', linestyle='--')
        plt.plot(range(len(trot_array[:, i]),), trot_array[:, i], label=f'trot:{i}')
        plt.legend()
    plt.show()