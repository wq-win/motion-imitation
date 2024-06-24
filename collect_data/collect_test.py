import copy
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
from matplotlib import pyplot as plt
import numpy as np
from collect_data import collect_oma_data_from_pma_12dim as codfp


p_position = []
p_direction = []
o_position = []
o_direction = []

JOINT_NUMS = codfp.JOINT_NUMS
TIMESTEP = codfp.TIMESTEP
LEN = codfp.PACE_LEN

pma = codfp.pma
pma_v, pma_v_norm, p_mass_weight = codfp.calculate_ring_velocity(pma)
p_position.append(pma)
p_direction.append(pma_v)

oma = codfp.oma
oma_v, oma_v_norm, o_mass_weight = codfp.calculate_ring_velocity(oma)
o_position.append(oma)
o_direction.append(oma_v)
def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
def ploter(position, direction, x=0, y=1, z=2, u=0, v=1, w=2):
    ax = plt.figure().add_subplot(projection='3d')
    X = position[:, x]
    Y = position[:, y]
    Z = position[:, z]
    U = direction[:, u]
    V = direction[:, v]
    W = direction[:, w]
    ax.quiver(X, Y, Z, U, V, W, length=TIMESTEP, normalize=False)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    # ax.set_title('Quiver Plot')
    set_axes_equal(ax)
    plt.show()


# test pma
p_point = codfp.sample_random_point(pma)  
for i in range(100):
    p_position.append(copy.deepcopy(p_point))
    normal_direction = codfp.calculate_point_normal_direction(pma, p_point) 
    normal_displacement = codfp.repulse(normal_direction, pma, p_point)
    tangent_displacement = codfp.calculate_point_tangent_velocity(pma, p_point)
    displacement =  tangent_displacement + normal_displacement
    displacement = codfp.calculate_point_displacement(pma, p_point, displacement)
    
    p_point += displacement * TIMESTEP
    p_direction.append(displacement)
   
p_position = np.vstack(p_position)
p_direction = np.vstack(p_direction)    
ploter(p_position, p_direction,)


# # test oma
# o_point = codfp.sample_random_point(oma)
# for i in range(100):
#     o_position.append(copy.deepcopy(o_point))
#     normal_direction, point2ring_displacemnt_norm, point2ring_nearest_index, ring_nearest_index_v, point2ring_nearest_displacement, distances_flag, = codfp.calculate_point_normal_direction(oma, o_point)
#     ma_v, ma_v_norm, ma_weight = codfp.calculate_ring_velocity(oma)
#     normal_displacement = codfp.repulse(normal_direction, point2ring_displacemnt_norm, ma_weight)
#     o_direction.append(normal_displacement)
#     o_point += normal_displacement * TIMESTEP
# o_position = np.vstack(o_position)
# o_direction = np.vstack(o_direction)
# ploter(o_position, o_direction,)

# two_array = np.vstack((pma, oma))
# two_array_v = np.vstack((pma_v, oma_v))
# ploter(two_array, two_array_v)

# codfp.trajactory_ploter(two_array, two_array_v,
#                         index_range, JOINT_NUMS, color_array, 0,1,2,0,1,2)
# codfp.trajactory_ploter(two_array, two_array_v,
#                         index_range, JOINT_NUMS, color_array, 3,4,5,3,4,5)
# codfp.trajactory_ploter(two_array, two_array_v,
#                         index_range, JOINT_NUMS, color_array, 6,7,8,6,7,8)
# codfp.trajactory_ploter(two_array, two_array_v,
#                         index_range, JOINT_NUMS, color_array, 9,10,11,9,10,11)
# plt.figure()
# for i in range(12):
#     plt.subplot(4, 3, i+1)
#     plt.plot(range(len(pma[:, i]),), pma[:, i], label=f'pma:{i}', linestyle='-')
#     plt.plot(range(len(oma[:, i]),), oma[:, i], label=f'oma:{i}', linestyle='--')
#     plt.legend()
# plt.show()
