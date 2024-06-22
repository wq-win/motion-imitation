import copy
import os
import pickle
import time
from matplotlib import pyplot as plt
import numpy as np
import tqdm
np.seterr(all='raise')
np.random.seed(0)

NOWTIME = time.strftime("%m_%d", time.localtime())

CONSTAN_FACTOR = 2000 
SAMPLE_POINT_NUMS = 10 # 1e5
SAMPLE_POINT_NUMS = int(SAMPLE_POINT_NUMS)
ITER_TIMES = 100
TIMESTEP = 1 / 30
JOINT_NUMS = 12
JOINT_INDEX_START = 7
DISTANCEMENT_THRESHOLD = 2
input_list = []
output_list = []
color_list = []

pace = [
    [0.00000, 0.00000, 0.43701, 0.49491, 0.53393, 0.49912, 0.46997, -0.12721, 0.07675, -0.95545, -0.25301, 0.18682,
     -1.14403, -0.19362, 0.14030, -0.77823, -0.09528, 0.05437, -0.97596],
    [0.01641, 0.00223, 0.43771, 0.48959, 0.53669, 0.50119, 0.47018, -0.12680, 0.11820, -0.94606, -0.28172, 0.03357,
     -1.16456, -0.20247, 0.17747, -0.77104, -0.09744, -0.05174, -0.93399],
    [0.03278, 0.00476, 0.43896, 0.48274, 0.53845, 0.50530, 0.47084, -0.12518, 0.15584, -0.92492, -0.30683, -0.11684,
     -1.15057, -0.21314, 0.22216, -0.76688, -0.10566, -0.14981, -0.88721],
    [0.04882, 0.00706, 0.44055, 0.47656, 0.53895, 0.50939, 0.47217, -0.12177, 0.19077, -0.89664, -0.31713, -0.25529,
     -1.10780, -0.22449, 0.26890, -0.75917, -0.11668, -0.23940, -0.83319],
    [0.06588, 0.00883, 0.44210, 0.47093, 0.53858, 0.51304, 0.47428, -0.12087, 0.22330, -0.86296, -0.31269, -0.35342,
     -1.05107, -0.23663, 0.31297, -0.74403, -0.12989, -0.31593, -0.77429],
    [0.08286, 0.01033, 0.44397, 0.46634, 0.53774, 0.51543, 0.47717, -0.12319, 0.25043, -0.82652, -0.29625, -0.41874,
     -0.97904, -0.24941, 0.35253, -0.72424, -0.13472, -0.37088, -0.71971],
    [0.09884, 0.01171, 0.44605, 0.46371, 0.53747, 0.51525, 0.48022, -0.12780, 0.26864, -0.79123, -0.27806, -0.47452,
     -0.91527, -0.26153, 0.38053, -0.69634, -0.13017, -0.40471, -0.67348],
    [0.11564, 0.01337, 0.44783, 0.46172, 0.53682, 0.51397, 0.48422, -0.13502, 0.28660, -0.75117, -0.28068, -0.51479,
     -0.83098, -0.27053, 0.40688, -0.66852, -0.11715, -0.41160, -0.63590],
    [0.13247, 0.01557, 0.44866, 0.46017, 0.53653, 0.51145, 0.48867, -0.14270, 0.30399, -0.71655, -0.21607, -0.52903,
     -0.77176, -0.27539, 0.43360, -0.64668, -0.10623, -0.37913, -0.63582],
    [0.14967, 0.01760, 0.44808, 0.45870, 0.53617, 0.50991, 0.49205, -0.14760, 0.32861, -0.68749, -0.16653, -0.50534,
     -0.73515, -0.27867, 0.47223, -0.65260, -0.10817, -0.30617, -0.68954],
    [0.16688, 0.01936, 0.44679, 0.45801, 0.53697, 0.50846, 0.49332, -0.14926, 0.35277, -0.66859, -0.14587, -0.46223,
     -0.71030, -0.27905, 0.52277, -0.69369, -0.10386, -0.25300, -0.72486],
    [0.18457, 0.02113, 0.44491, 0.45884, 0.53861, 0.50792, 0.49131, -0.14788, 0.37881, -0.65994, -0.12952, -0.41299,
     -0.68721, -0.26795, 0.55542, -0.74544, -0.10247, -0.19779, -0.76080],
    [0.20315, 0.02255, 0.44338, 0.46159, 0.54103, 0.50733, 0.48666, -0.14168, 0.40260, -0.65184, -0.16240, -0.34246,
     -0.76043, -0.24415, 0.54899, -0.78040, -0.10484, -0.15274, -0.78151],
    [0.22288, 0.02351, 0.44195, 0.46521, 0.54355, 0.50682, 0.48089, -0.13629, 0.42854, -0.65674, -0.16512, -0.29743,
     -0.79699, -0.22199, 0.52110, -0.82139, -0.10857, -0.11585, -0.79425],
    [0.24349, 0.02420, 0.44024, 0.46907, 0.54564, 0.50727, 0.47426, -0.12753, 0.44960, -0.68466, -0.17700, -0.23585,
     -0.84852, -0.21056, 0.47836, -0.87137, -0.11339, -0.07894, -0.80477],
    [0.26441, 0.02473, 0.43823, 0.47325, 0.54703, 0.50712, 0.46865, -0.11968, 0.45798, -0.76842, -0.19037, -0.17536,
     -0.89554, -0.20650, 0.44393, -0.92752, -0.11522, -0.04086, -0.81465],
    [0.28558, 0.02464, 0.43620, 0.47835, 0.54689, 0.50616, 0.46464, -0.12197, 0.43800, -0.86342, -0.19421, -0.11435,
     -0.93678, -0.19062, 0.40079, -0.97485, -0.11837, -0.00403, -0.82084],
    [0.30563, 0.02398, 0.43551, 0.48404, 0.54571, 0.50448, 0.46196, -0.12253, 0.41137, -0.95532, -0.19119, -0.06019,
     -0.96439, -0.17122, 0.33514, -1.00492, -0.11824, 0.02713, -0.81663],
    [0.32565, 0.02340, 0.43543, 0.48863, 0.54314, 0.50286, 0.46192, -0.12639, 0.37667, -1.06982, -0.18340, -0.00535,
     -0.97909, -0.16186, 0.24489, -1.00477, -0.11752, 0.06483, -0.81530],
    [0.34513, 0.02264, 0.43600, 0.49248, 0.53972, 0.50166, 0.46315, -0.13757, 0.29075, -1.15527, -0.17385, 0.04504,
     -0.97993, -0.16089, 0.14852, -0.98629, -0.11866, 0.10202, -0.80805],
    [0.36352, 0.02156, 0.43678, 0.49651, 0.53506, 0.50047, 0.46553, -0.14822, 0.15752, -1.19552, -0.16643, 0.09010,
     -0.97221, -0.16121, 0.04991, -0.95571, -0.12166, 0.14042, -0.79969],
    [0.38133, 0.01988, 0.43839, 0.50053, 0.52944, 0.49854, 0.46971, -0.15608, 0.00492, -1.19387, -0.15410, 0.13039,
     -0.95309, -0.16926, -0.05029, -0.90903, -0.12653, 0.17847, -0.78722],
    [0.39841, 0.01837, 0.44059, 0.50384, 0.52281, 0.49739, 0.47479, -0.15983, -0.13783, -1.15844, -0.14303, 0.16753,
     -0.92657, -0.18165, -0.14768, -0.85218, -0.13280, 0.21770, -0.77474],
    [0.41513, 0.01669, 0.44294, 0.50649, 0.51629, 0.49694, 0.47954, -0.15788, -0.25191, -1.09422, -0.13183, 0.19619,
     -0.88943, -0.19374, -0.23443, -0.78796, -0.14329, 0.25547, -0.75380],
    [0.43177, 0.01473, 0.44521, 0.50907, 0.51037, 0.49701, 0.48306, -0.15548, -0.33574, -1.01839, -0.12400, 0.22032,
     -0.84860, -0.20385, -0.30441, -0.72873, -0.15418, 0.28667, -0.72651],
    [0.44802, 0.01314, 0.44760, 0.51097, 0.50622, 0.49819, 0.48420, -0.15480, -0.40127, -0.95462, -0.12135, 0.23766,
     -0.80623, -0.20927, -0.35212, -0.67100, -0.16640, 0.31242, -0.69448],
    [0.46489, 0.01140, 0.45021, 0.51169, 0.50451, 0.49981, 0.48356, -0.14707, -0.45308, -0.87697, -0.11954, 0.25095,
     -0.75704, -0.21183, -0.38106, -0.61525, -0.17028, 0.33364, -0.66014],
    [0.48202, 0.00922, 0.45259, 0.51313, 0.50273, 0.50274, 0.48085, -0.08334, -0.47429, -0.78665, -0.12094, 0.26006,
     -0.70915, -0.19570, -0.37232, -0.58446, -0.17731, 0.34955, -0.62396],
    [0.49978, 0.00692, 0.45301, 0.51399, 0.49995, 0.50649, 0.47889, -0.03442, -0.44226, -0.72261, -0.12344, 0.27515,
     -0.66283, -0.19484, -0.31002, -0.61017, -0.18530, 0.38462, -0.62481],
    [0.51793, 0.00522, 0.45155, 0.51416, 0.49704, 0.50996, 0.47805, -0.03327, -0.39805, -0.70482, -0.12444, 0.29801,
     -0.62361, -0.20098, -0.25119, -0.65274, -0.19097, 0.44064, -0.67343],
    [0.53748, 0.00395, 0.44913, 0.51411, 0.49529, 0.51164, 0.47813, -0.07110, -0.32111, -0.76216, -0.12382, 0.32680,
     -0.59657, -0.19979, -0.19735, -0.68530, -0.18261, 0.49555, -0.73822],
    [0.55659, 0.00280, 0.44697, 0.51485, 0.49644, 0.50985, 0.47805, -0.07549, -0.26493, -0.80317, -0.11877, 0.35467,
     -0.59078, -0.20209, -0.14052, -0.72329, -0.16125, 0.51232, -0.79071],
    [0.57577, 0.00167, 0.44507, 0.51650, 0.49783, 0.50696, 0.47789, -0.07131, -0.21348, -0.82662, -0.11386, 0.37332,
     -0.60466, -0.19963, -0.08971, -0.74502, -0.14228, 0.49408, -0.83879],
    [0.59497, 0.00154, 0.44312, 0.51800, 0.50013, 0.50330, 0.47774, -0.08576, -0.15562, -0.87133, -0.11535, 0.37462,
     -0.65372, -0.20273, -0.04893, -0.76230, -0.12839, 0.46938, -0.90473],
    [0.61400, 0.00191, 0.44160, 0.51843, 0.50324, 0.49974, 0.47774, -0.09484, -0.10177, -0.90486, -0.12731, 0.36711,
     -0.75677, -0.20671, -0.01317, -0.77205, -0.11790, 0.43685, -0.97024],
    [0.63303, 0.00233, 0.44009, 0.51746, 0.50557, 0.49713, 0.47905, -0.10425, -0.04639, -0.92873, -0.14730, 0.35852,
     -0.86271, -0.21267, 0.02886, -0.78466, -0.10501, 0.39079, -1.01703],
    [0.65155, 0.00263, 0.43886, 0.51586, 0.50831, 0.49492, 0.48016, -0.11146, 0.00547, -0.94503, -0.18348, 0.32590,
     -0.98252, -0.21663, 0.07145, -0.79413, -0.09216, 0.31946, -1.04176],
    [0.67005, 0.00126, 0.43824, 0.51299, 0.51174, 0.49342, 0.48113, -0.12047, 0.05387, -0.95210, -0.21892, 0.23998,
     -1.07604, -0.22485, 0.10828, -0.79239, -0.08403, 0.22582, -1.04134],
    [0.68773, 0.00000, 0.43701, 0.50903, 0.51581, 0.49242, 0.48203, -0.12785, 0.09815, -0.95073, -0.26299, 0.10340,
     -1.12756, -0.23415, 0.13683, -0.78085, -0.07723, 0.11886, -1.01564]
]
pace_len = len(pace)
pace_array = np.array(pace)
p_motor_angle = pace_array[:, JOINT_INDEX_START:JOINT_INDEX_START + JOINT_NUMS]
p_motor_angle_next = np.vstack((p_motor_angle[1:, :], p_motor_angle[:1, :]))
p_motor_angle_displacement = p_motor_angle_next - p_motor_angle
p_motor_angle_v = p_motor_angle_displacement / TIMESTEP
p_motor_angle_v_norm = np.linalg.norm(p_motor_angle_v, axis=1, keepdims=1)
mass_weight = p_motor_angle_v_norm / np.max(p_motor_angle_v_norm)
input_list.append(p_motor_angle)
output_list.append(p_motor_angle_v)
color_list.append(np.zeros((pace_len, 3)))


p_motor_angle_tile = np.tile(p_motor_angle, (int(SAMPLE_POINT_NUMS / 4), 1))
p_motor_angle_v_tile = np.tile(p_motor_angle_v, (int(SAMPLE_POINT_NUMS / 4), 1))
input_list.append(p_motor_angle_tile)
output_list.append(p_motor_angle_v_tile)
color_list.append(np.zeros((p_motor_angle_tile.shape[0], 3)))

def sigmoid(x : np.array):
    # return .5 * x
    return (1 / (1 + np.exp(-x)) - .5)
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


def sample_random_point():
    """
    min=a,max=b, 随机生成一个在[a-(b-a),b+(b-a)]内的12dim随机数
    """
    point = []
    for i in range(JOINT_NUMS):
        dim_min, dim_max = min(p_motor_angle[:, i]), max(p_motor_angle[:, i])
        point.append(np.random.uniform(2 * dim_min - dim_max, 2 * dim_max - dim_min))
    return point


def sample_random_point_pi():
    point = np.random.uniform(-np.pi, np.pi, size=JOINT_NUMS)
    return point


def calculate_point_normal_direction(point, mass_weight):
    point2ring_displacement = p_motor_angle - point
    point2ring_distances = np.linalg.norm(point2ring_displacement, axis=1, keepdims=True)
    # x^(1/6)
    # point2ring_distances = point2ring_distances**(1/6)
    point2ring_nearest_index_temp = np.argmin(point2ring_distances)
    if np.dot(point2ring_displacement[point2ring_nearest_index_temp],
              p_motor_angle_v[point2ring_nearest_index_temp]) < 0:
        point2ring_nearest_index = point2ring_nearest_index_temp + 1
        if point2ring_nearest_index >= pace_len:
            point2ring_nearest_index = 0
    else:
        point2ring_nearest_index = point2ring_nearest_index_temp
    ring_nearest_index_v = p_motor_angle_v[point2ring_nearest_index]
    ring_nearest_index_v_norm = np.linalg.norm(ring_nearest_index_v)
    distances_flag = point2ring_distances[point2ring_nearest_index] < p_motor_angle_v_norm[point2ring_nearest_index] * TIMESTEP * DISTANCEMENT_THRESHOLD
    if distances_flag:
        displacement_on_ring_projection = np.dot(point2ring_displacement[point2ring_nearest_index],
                                                 ring_nearest_index_v) / ring_nearest_index_v_norm
        point2ring_normal_vector = point2ring_displacement[
                                       point2ring_nearest_index] - displacement_on_ring_projection * ring_nearest_index_v / ring_nearest_index_v_norm
        testa = np.dot(point2ring_normal_vector, ring_nearest_index_v)
        assert np.abs(testa) < 0.01, 'not normal %f' % (testa)
        direction = point2ring_normal_vector / np.linalg.norm(point2ring_normal_vector)
    else:
        forces = mass_weight * point2ring_displacement / (point2ring_distances ** 2)
        force = np.sum(forces, axis=0) / forces.shape[0]
        direction = force / np.linalg.norm(force)
    return direction, point2ring_distances, point2ring_nearest_index, ring_nearest_index_v, point2ring_displacement[
        point2ring_nearest_index], distances_flag

def calculate_point_displacement(distances_flag, ring_nearest_index_v, displacement):
    displacement_normalize = displacement / np.linalg.norm(displacement)
    ring_nearest_index_v_norm = np.linalg.norm(ring_nearest_index_v)
    if True:
        return displacement_normalize * ring_nearest_index_v_norm
    else:
        return displacement
def calculate_point_tangent_velocity(distances, point2ring_nearest_index, ring_nearest_index_v,
                                     point2ring_nearest_displacement, decay=10):  # 200
    # for i, distance in enumerate(distances):
    #     if distance < 1e-6:
    #         return p_motor_angle_v[i]

    if distances[point2ring_nearest_index] < 1e-4:
        return p_motor_angle_v[point2ring_nearest_index]
    # distance = np.sum(1 / (distances ), axis=0)
    # forces = p_motor_angle_v / (distances) / (distances**2 + 1)
    # force = np.sum(forces, axis=0)
    # velocity = force / distance

    forces = p_motor_angle_v / (distances ** 2) / (decay * distances + 1)
    force = np.sum(forces, axis=0)
    distance = np.sum(1 / (distances ** 2), axis=0)
    velocity = force / distance
    if np.linalg.norm(point2ring_nearest_displacement) < p_motor_angle_v_norm[point2ring_nearest_index] * TIMESTEP * DISTANCEMENT_THRESHOLD:
        ring_nearest_index_v_norm = np.linalg.norm(ring_nearest_index_v)
        t_velocity = (np.dot(velocity, ring_nearest_index_v) * ring_nearest_index_v /
                      ring_nearest_index_v_norm ** 2)
    else:
        t_velocity = velocity
    return t_velocity


def repulse(direction, distances, mass_weight):
    ring_point_nums = distances.shape[0]
    speed = CONSTAN_FACTOR / np.sum(1 / distances * mass_weight) / ring_point_nums
    displacement = direction * speed
    return displacement


def trajactory_ploter(position, arrow, index_range=(0, -1), dim=3, color_array=None, x=0, y=1, z=2, u=0, v=1, w=2):
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
        ax.quiver(X, Y, Z, U, V, W, color=color_array, normalize=False, length=TIMESTEP)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title(f'{dim} dimension, x={x}, y={y}, z={z}')
    set_axes_equal(ax)

    plt.show()


if __name__ == '__main__':
    start_index = sum(len(an_output) for an_output in output_list)
    for _ in tqdm.tqdm(range(int(SAMPLE_POINT_NUMS / 2))):
        point = sample_random_point()
        for i in range(ITER_TIMES):
            input_list.append(copy.deepcopy(point))
            normal_direction, point2ring_distances, point2ring_nearest_index, ring_nearest_index_v, point2ring_nearest_displacement, distances_flag = calculate_point_normal_direction(
                point, mass_weight)
            
            normal_displacement = repulse(normal_direction, point2ring_distances, mass_weight)
            tangent_displacement = calculate_point_tangent_velocity(point2ring_distances, point2ring_nearest_index,
                                                                    ring_nearest_index_v,
                                                                    point2ring_nearest_displacement)
            displacement =  tangent_displacement + normal_displacement #+  # + t_displacement
            displacement = calculate_point_displacement(distances_flag, ring_nearest_index_v, displacement)
            point += displacement * TIMESTEP
            output_list.append(displacement)
        color = np.ones((ITER_TIMES, 3))
        color[:, 1] = np.linspace(0.8, 0, ITER_TIMES)
        color[:, 0] = np.linspace(0.8, 0, ITER_TIMES)
        # color[:, 2] = np.linspace(0.8, 0, ITER_TIMES)
        color_list.append(color)

    for _ in tqdm.tqdm(range(int(SAMPLE_POINT_NUMS / 4))):
        point = sample_random_point_pi()
        for i in range(ITER_TIMES):
            input_list.append(copy.deepcopy(point))
            normal_direction, point2ring_distances, point2ring_nearest_index, ring_nearest_index_v, point2ring_nearest_displacement, distances_flag = calculate_point_normal_direction(
                point, mass_weight)
            normal_displacement = repulse(normal_direction, point2ring_distances, mass_weight)
            tangent_displacement = calculate_point_tangent_velocity(point2ring_distances, point2ring_nearest_index,
                                                                    ring_nearest_index_v,
                                                                    point2ring_nearest_displacement)
            displacement = tangent_displacement + normal_displacement  # + t_displacement
            displacement = calculate_point_displacement(distances_flag, ring_nearest_index_v, displacement)
            point += displacement * TIMESTEP
            output_list.append(displacement)
        color = np.ones((ITER_TIMES, 3))
        color[:, 1] = np.linspace(0.8, 0, ITER_TIMES)
        color[:, 2] = np.linspace(0.8, 0, ITER_TIMES)
        color_list.append(color)

    input_array = np.vstack(input_list)
    output_array = np.vstack(output_list)
    print(np.max(np.linalg.norm(output_array[:start_index], axis=1)))
    print(np.max(np.linalg.norm(output_array[start_index:], axis=1)))
    color_array = np.vstack(color_list)

    trajactory_ploter(input_array, output_array, index_range=[0000, 1000], dim=JOINT_NUMS, color_array=color_array, x=0,
                      y=1, z=2, u=0, v=1, w=2)
    trajactory_ploter(input_array, output_array, index_range=[0000, 1000], dim=JOINT_NUMS, color_array=color_array, x=3,
                      y=4, z=5, u=3, v=4, w=5)
    trajactory_ploter(input_array, output_array, index_range=[0000, 1000], dim=JOINT_NUMS, color_array=color_array, x=6,
                      y=7, z=8, u=6, v=7, w=8)
    trajactory_ploter(input_array, output_array, index_range=[0000, 1000], dim=JOINT_NUMS, color_array=color_array, x=9,
                      y=10, z=11, u=9, v=10, w=11)
    allresult = {'input': input_array, 'output': output_array}
    file_path = f'dataset/save_data_V{NOWTIME}_{SAMPLE_POINT_NUMS}_{ITER_TIMES}.pkl'
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(file_path, 'wb') as f:
        pickle.dump(allresult, f)
