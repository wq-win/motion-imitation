import copy
import pickle
from matplotlib import pyplot as plt
import numpy as np


def oma_to_a(oma: np.ndarray) -> np.ndarray:
    action = copy.deepcopy(oma)
    if len(oma.shape) > 1 and oma.shape[0] > 1:
        # action[:, np.array([0, 6])] = -oma[:, np.array([0, 6])]
        action[:, np.array([1, 4, 7, 10])] -= 0.67
        action[:, np.array([2, 5, 8, 11])] -= -1.25
    else:
        # action[np.array([0, 6])] = -oma[np.array([0,6])]
        action[np.array([1, 4, 7, 10])] -= 0.67  
        action[np.array([2, 5, 8, 11])] -= -1.25  
    return action    


if __name__ == '__main__':
    with open('trot/dataset/collect_a_oma_data.pkl', 'rb') as f:
        allresult = pickle.load(f)
        
    o = allresult['o']
    a = allresult['a']
    o = np.array(o)
    a = np.array(a)
    oma = o[:, 48:60]
    oma_add_offset = oma_to_a(oma)
    plt.figure()
    for i in range(12):
        plt.subplot(4, 3, i+1)
        plt.plot(range(len(oma_add_offset[4:, i]),), oma_add_offset[4:, i], label=f'o:{i}', linestyle='-')
        plt.plot(range(len(a[:, i]),), a[:, i], label=f'a:{i}', linestyle='--')
        plt.legend()
    plt.show()