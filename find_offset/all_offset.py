import numpy as np


def oma_to_pma(oma):
    oma[np.array([0, 6])] = -oma[np.array([0,6])]
    oma[np.array([1, 4, 7, 10])] -= 0.6
    oma[np.array([2, 5, 8, 11])] -= -0.66
    pma = oma
    return pma


def oma_to_right_action(oma):
    oma[np.array([0, 3, 6, 9])] = -oma[np.array([0, 3, 6, 9])]  # 0,6 -= -0.00325897 ; 3,9 -= 0.0034551
    oma[np.array([1, 4, 7, 10])] -= 0.67  # 0.6742553
    oma[np.array([2, 5, 8, 11])] -= -1.25 # -1.25115246
    right_action = oma
    return right_action

def offset_to_right_action(oma):
    oma[np.array([0, 6])] -= -0.00325897
    oma[np.array([3, 9])] -= 0.0034551
    oma[np.array([1, 4, 7, 10])] -= 0.6742553
    oma[np.array([2, 5, 8, 11])] -= -1.25115246
    return oma

def test_to_right_action(oma):
    oma[np.array([1, 4, 7, 10])] -= 0.67
    oma[np.array([2, 5, 8, 11])] -= -1.25
    return oma