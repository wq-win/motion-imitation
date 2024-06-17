import pickle

from matplotlib import pyplot as plt
import numpy as np


num_joints = 12
ITER_TIMES = 50
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

def trajactory_ploter(position, arrow, index_range=(0.-1), dim=3, color_array=None, x=0, y=1, z=2, u=0, v=1, w=2):
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
        ax.quiver(X, Y, Z, U, V, W, normalize=True, length=0.03)
    else:
        ax.quiver(X, Y, Z, U, V, W, color=color_array, normalize=True, length=0.01)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('%d dimension'%(dim))
    set_axes_equal(ax)
    
    plt.show()  

with open('dataset/save_data_V4_100000_10.pkl', 'rb') as f:
            allresult = pickle.load(f)

input = np.array(allresult['input'])
output = np.array(allresult['output'])
print(input.shape)
for i in range(input.shape[0]//1000 // 20):
    trajactory_ploter(input, output, index_range=[i * 1000, (i + 1) * 1000], dim=num_joints, color_array=None,  x=0, y=1, z=2, u=0, v=1, w=2)
# trajactory_ploter(input, output, index_range=[0, 1000], dim=num_joints, color_array=None,  x=0, y=1, z=2, u=0, v=1, w=2)
