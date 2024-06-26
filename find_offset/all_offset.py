import copy
import numpy as np

def pma_to_oma(pma: np.ndarray) -> np.ndarray:
    assert pma.shape[1] == 12, 'The dimension of pma is not 12'
    oma = copy.deepcopy(pma)
    if pma.shape[0] > 1:
        oma[:, np.array([0, 6])] = -pma[:, np.array([0, 6])]
        oma[:, np.array([1, 4, 7, 10])] -= -0.6
        oma[:, np.array([2, 5, 8, 11])] -= 0.66
    else:
        oma[np.array([0, 6])] = -pma[np.array([0, 6])]
        oma[np.array([1, 4, 7, 10])] -= -0.6
        oma[np.array([2, 5, 8, 11])] -= 0.66
    return oma


def pma_to_oma_3dim(pma: np.ndarray) -> np.ndarray:
    assert pma.shape[1] == 3, 'The dimension of pma is not 3'
    oma = copy.deepcopy(pma)
    if pma.shape[0] > 1:
        oma[:, np.array([0])] = -pma[:, np.array([0])]
        oma[:, np.array([1])] -= -0.6
        oma[:, np.array([2])] -= 0.66
    else:
        oma[np.array([0])] = -pma[np.array([0])]
        oma[np.array([1])] -= -0.6
        oma[np.array([2])] -= 0.66
    return oma


# TODO:fix the bug
def oma_to_pma(oma):
    pma = copy.deepcopy(oma)
    pma[:, np.array([0, 6])] = -oma[:, np.array([0, 6])]
    pma[:, np.array([1, 4, 7, 10])] += -0.6
    pma[:, np.array([2, 5, 8, 11])] += 0.66
    return pma
def oma_to_right_action(oma):
    action = copy.deepcopy(oma)
    # 0,6 -= -0.00325897 ; 3,9 -= 0.0034551
    action[np.array([0, 3, 6, 9])] = -oma[np.array([0, 3, 6, 9])]
    action[np.array([1, 4, 7, 10])] -= 0.67  # 0.6742553
    action[np.array([2, 5, 8, 11])] -= -1.25  # -1.25115246
    return action


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





def error_between_target_and_result(o):
    """
    target motorangle is o[12:24]=env.step input action
    result motorangle is o[48:60]=current observation motorangle
    """
    error = o[12:24] - o[48:60]
    return error
