from matplotlib import pyplot as plt
import numpy as np

A = np.array([-1, 1])
B = np.array([1, 1])
C = np.array([0, 0])

AUV = B - A
# BUV = [0, 0]
CUV = A - C
C_projection2AB = np.dot(CUV, AUV) / np.dot(AUV, AUV) * AUV
C_normal = CUV - C_projection2AB

fig = plt.figure()
ax = fig.add_subplot(111)
ax.quiver(A[0], A[1], AUV[0], AUV[1], angles='xy', scale_units='xy', scale=1, color='r')
# ax.quiver(B[0], B[1], BUV[0], BUV[1], angles='xy', scale_units='xy', scale=1, color='r')
ax.quiver(C[0], C[1], CUV[0], CUV[1], angles='xy', scale_units='xy', scale=1, color='b')
# ax.quiver(C[0], C[1], C_projection2AB[0], C_projection2AB[1], angles='xy', scale_units='xy', scale=1, color='r')
ax.quiver(C[0], C[1], C_normal[0], C_normal[1], angles='xy', scale_units='xy', scale=1, color='g')
ax.axis([-1.5, 1.5,-0.5,1.5])
plt.show()
