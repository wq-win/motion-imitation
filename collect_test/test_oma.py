import os
import inspect
import pickle
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import numpy as np
from matplotlib import pyplot as plt
from collect import collect_oma_data


TIMESTEP = collect_oma_data.TIMESTEP


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
    plot_radius = 0.5 * max([x_range, y_range, z_range])

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


def trajactory_ploter(position, arrow, index_range=(0, -1), dim=12, color_array=None, x=0, y=1, z=2, u=0, v=1, w=2):
    ax = plt.figure().add_subplot(projection='3d')

    # Make the grid
    X = position[index_range[0]:index_range[1], x]
    Y = position[index_range[0]:index_range[1], y]
    Z = position[index_range[0]:index_range[1], z]

    # Make the direction data for the arrows
    U = arrow[index_range[0]:index_range[1], u]
    V = arrow[index_range[0]:index_range[1], v]
    W = arrow[index_range[0]:index_range[1], w]

    if color_array is None:
        ax.quiver(X, Y, Z, U, V, W, normalize=False, length=TIMESTEP)
    else:
        ax.quiver(X, Y, Z, U, V, W, color=color_array,
                  normalize=False, length=TIMESTEP)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title(f'{dim} dimension, x={x}, y={y}, z={z}')
    set_axes_equal(ax)

    plt.show()


if __name__ == '__main__':
    
    with open('collect_dataset/oma_data_V_06_26_4_100.pkl', 'rb') as f:
        allresult = pickle.load(f)
    input_array = allresult['input']
    output_array = allresult['output']
    ITER_TIMES = allresult['ITER_TIMES']
    SAMPLE_POINT_NUMS = allresult['SAMPLE_POINT_NUMS']
    
    # ploter(input_array, output_array)
    SUM_NUMS = ITER_TIMES * SAMPLE_POINT_NUMS
    color = np.ones((SUM_NUMS, 3))
    for i in range(SAMPLE_POINT_NUMS):
        color[i * ITER_TIMES: (i + 1) * ITER_TIMES, 0] = np.linspace(0.8, 0, ITER_TIMES)
        color[i * ITER_TIMES: (i + 1) * ITER_TIMES, 1] = np.linspace(0.8, 0, ITER_TIMES)
        # color[i * ITER_TIMES: (i + 1) * ITER_TIMES, 2] = np.linspace(0.8, 0, ITER_TIMES)
    if  SUM_NUMS > 1e3:
        index_range = (0, 1000)
    else:
        index_range = (0, -1)
    trajactory_ploter(input_array, output_array, index_range=index_range, color_array=color)
    # plt.figure()
    # for i in range(12):
    #     plt.subplot(4, 3, i+1)
    #     plt.plot(range(len(pma[:, i]),), pma[:, i], label=f'pma:{i}', linestyle='-')
    #     plt.legend()
    # plt.show()