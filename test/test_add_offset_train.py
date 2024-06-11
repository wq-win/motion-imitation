import pickle

import numpy as np


with open('dataset/o_a_collect_nums_1000.pkl', 'rb') as f:
            allresult = pickle.load(f)
o = np.array(allresult['o'], dtype=float)

motor_angle = o[:, 48:60]
print(motor_angle[0])
motor_angle[:, np.array([0, 6])] = -motor_angle[:, np.array([0, 6])]
motor_angle[:, np.array([1, 4, 7, 10])] -= .6
motor_angle[:, np.array([2, 5, 8, 11])] -= -.66

print(motor_angle[0])
