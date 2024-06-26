
import copy

import numpy as np


def oma_to_right_action(oma:np.ndarray) -> np.ndarray:
    action = copy.deepcopy(oma)
    action[np.array([0, 3, 6, 9])] = -oma[np.array([0, 3, 6, 9])]  
    action[np.array([1, 4, 7, 10])] -= 0.67
    action[np.array([2, 5, 8, 11])] -= -1.25
    return action


def error_between_target_and_result(o:np.ndarray, ignore_hip=False) -> np.ndarray:
    """
    target motorangle is o[12:24]=env.step input action
    result motorangle is o[48:60]=current observation motor angle
    """
    error = o[12:24] - o[48:60]
    if ignore_hip:
        error[np.array([0, 3, 6, 9])]=0
    return error


