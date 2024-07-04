"""
ma means motor angle.
Subtracting two points gives displacement. Use displacement to represent.
displacement length is the distance between two points. Use displacement_norm to represent.
displacement divided by distance is normalized displacement. Use displacement_normalized to represent.
displacement divided by time is velocity. Use v to represent.
velocity length is the speed of the point. Use v_norm to represent.
velocity divided by speed is normalized velocity. Use v_normalized to represent.
The speed weight provided by each ring point is ma_weight.Here, the maximum weight is 1. So it's speed divided by maximum speed.
"""
import copy
import pickle
import time
from matplotlib import pyplot as plt
import numpy as np
import os
import inspect

import tqdm
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)


NOWTIME = time.strftime("%m_%d", time.localtime())
spin = [
  [0.00000, 0.00000, 0.44976, 0.54057, 0.44882, 0.49571, 0.51052, 0.06576, -0.04242, -0.82964, -0.28412, 0.03276, -0.85943, -0.05603, -0.06834, -0.80993, 0.18463, 0.14715, -0.92528],
  [0.00170, -0.00521, 0.44895, 0.53396, 0.46602, 0.50437, 0.49328, 0.08427, -0.04042, -0.84964, -0.26965, 0.02893, -0.90262, -0.08150, -0.08507, -0.79410, 0.11790, 0.12385, -0.94709],
  [0.00294, -0.01037, 0.44906, 0.52667, 0.48140, 0.51294, 0.47724, 0.10163, -0.04171, -0.86061, -0.24761, 0.02580, -0.94555, -0.10571, -0.10000, -0.77467, 0.05056, 0.09522, -0.94786],
  [0.00427, -0.01548, 0.44956, 0.51835, 0.49567, 0.52238, 0.46125, 0.12089, -0.04302, -0.86683, -0.21255, 0.03453, -0.99109, -0.12795, -0.11315, -0.75647, -0.01603, 0.06023, -0.92879],
  [0.00526, -0.02026, 0.45050, 0.50886, 0.50938, 0.53230, 0.44525, 0.14275, -0.04471, -0.86730, -0.16893, 0.04941, -1.04211, -0.14831, -0.12830, -0.73635, -0.07620, 0.02178, -0.89569],
  [0.00660, -0.02422, 0.45199, 0.49779, 0.52177, 0.54282, 0.43048, 0.16632, -0.04216, -0.85923, -0.11659, 0.07414, -1.08248, -0.16676, -0.14716, -0.70362, -0.13189, -0.00839, -0.86821],
  [0.00816, -0.02738, 0.45291, 0.48520, 0.53279, 0.55408, 0.41679, 0.18885, -0.03685, -0.84672, -0.06279, 0.11092, -1.12056, -0.18593, -0.15269, -0.68631, -0.16323, -0.03373, -0.84157],
  [0.00912, -0.02982, 0.45334, 0.47064, 0.54404, 0.56518, 0.40384, 0.21070, -0.03267, -0.83000, -0.01177, 0.15189, -1.14205, -0.20995, -0.14839, -0.68509, -0.16257, -0.06147, -0.79858],
  [0.00959, -0.03131, 0.45411, 0.45509, 0.55530, 0.57608, 0.39072, 0.23517, -0.02957, -0.80754, 0.03302, 0.19194, -1.11850, -0.23488, -0.15766, -0.67462, -0.16007, -0.07681, -0.75131],
  [0.00820, -0.03198, 0.45512, 0.43882, 0.56589, 0.58720, 0.37737, 0.26272, -0.03132, -0.77755, 0.06785, 0.22387, -1.05869, -0.25299, -0.15864, -0.68217, -0.14566, -0.08876, -0.72233],
  [0.00543, -0.03250, 0.45493, 0.42083, 0.57510, 0.59930, 0.36471, 0.29036, -0.03125, -0.74467, 0.09422, 0.25073, -0.97785, -0.27355, -0.11169, -0.74634, -0.12336, -0.08627, -0.72625],
  [0.00266, -0.03330, 0.45293, 0.40085, 0.58320, 0.61230, 0.35254, 0.31519, -0.02544, -0.71341, 0.11186, 0.27263, -0.88991, -0.28671, -0.04936, -0.81543, -0.11084, -0.07331, -0.73991],
  [-0.00085, -0.03443, 0.45017, 0.37994, 0.59053, 0.62475, 0.34147, 0.33621, -0.01982, -0.68099, 0.12074, 0.28914, -0.79551, -0.28361, 0.02722, -0.89477, -0.09932, -0.06020, -0.76248],
  [-0.00634, -0.03619, 0.44723, 0.36039, 0.59826, 0.63447, 0.33113, 0.35389, -0.02085, -0.65397, 0.11318, 0.28671, -0.70350, -0.26150, 0.10645, -1.00152, -0.07910, -0.05584, -0.78561],
  [-0.01309, -0.03839, 0.44395, 0.34218, 0.60757, 0.64119, 0.32037, 0.37007, -0.02580, -0.63565, 0.09700, 0.27234, -0.64727, -0.21539, 0.18061, -1.10211, -0.05350, -0.06842, -0.79983],
  [-0.01936, -0.04049, 0.44018, 0.32374, 0.61866, 0.64630, 0.30781, 0.38802, -0.02803, -0.62488, 0.08022, 0.27466, -0.67156, -0.14826, 0.23933, -1.17471, -0.02692, -0.09472, -0.80488],
  [-0.02494, -0.04317, 0.43729, 0.30436, 0.63159, 0.64990, 0.29342, 0.40442, -0.03224, -0.61571, 0.07788, 0.26924, -0.73006, -0.07056, 0.27236, -1.20235, -0.00018, -0.13179, -0.79824],
  [-0.02932, -0.04633, 0.43552, 0.28442, 0.64515, 0.65290, 0.27676, 0.42248, -0.03499, -0.60695, 0.07813, 0.25674, -0.76520, 0.01565, 0.25803, -1.16953, 0.02243, -0.17780, -0.77968],
  [-0.03184, -0.04887, 0.43470, 0.26449, 0.65781, 0.65627, 0.25815, 0.44312, -0.03093, -0.60124, 0.07032, 0.24734, -0.80087, 0.08361, 0.22405, -1.09286, 0.03924, -0.22582, -0.75262],
  [-0.03289, -0.05072, 0.43489, 0.24439, 0.66848, 0.66046, 0.23916, 0.46143, -0.02316, -0.59879, 0.05933, 0.23618, -0.82066, 0.12051, 0.20659, -0.99043, 0.05365, -0.26812, -0.72343],
  [-0.03320, -0.05198, 0.43517, 0.22452, 0.67589, 0.66649, 0.22036, 0.47541, -0.01056, -0.60693, 0.04476, 0.23091, -0.82830, 0.13333, 0.21474, -0.89054, 0.06329, -0.30053, -0.70281],
  [-0.03402, -0.05331, 0.43476, 0.20529, 0.67901, 0.67509, 0.20263, 0.47786, 0.00542, -0.62423, 0.02775, 0.23387, -0.82772, 0.12607, 0.23394, -0.85245, 0.06711, -0.31594, -0.69964],
  [-0.03545, -0.05442, 0.43341, 0.18714, 0.67843, 0.68572, 0.18574, 0.46840, 0.02541, -0.64862, 0.00692, 0.24142, -0.82339, 0.09073, 0.27650, -0.88971, 0.06534, -0.31130, -0.71662],
  [-0.03754, -0.05531, 0.43273, 0.17030, 0.67675, 0.69595, 0.16927, 0.44696, 0.05146, -0.68563, -0.01593, 0.24399, -0.81205, 0.03948, 0.28861, -0.91399, 0.06061, -0.30093, -0.73706],
  [-0.04077, -0.05732, 0.43413, 0.15545, 0.67608, 0.70374, 0.15329, 0.41601, 0.07273, -0.73896, -0.03993, 0.23376, -0.79742, 0.00111, 0.28856, -0.92632, 0.05428, -0.30352, -0.73901],
  [-0.04404, -0.06019, 0.43745, 0.14245, 0.67675, 0.70876, 0.13921, 0.37668, 0.07835, -0.79680, -0.05741, 0.21535, -0.77849, -0.02091, 0.29496, -0.92252, 0.05020, -0.31831, -0.72156],
  [-0.04682, -0.06286, 0.44148, 0.12972, 0.67807, 0.71234, 0.12635, 0.32881, 0.07527, -0.85365, -0.07128, 0.19468, -0.76047, -0.03238, 0.30372, -0.90682, 0.04994, -0.33116, -0.70408],
  [-0.04911, -0.06459, 0.44379, 0.11697, 0.67839, 0.71628, 0.11422, 0.27572, 0.07319, -0.91204, -0.08422, 0.18025, -0.74867, -0.04532, 0.31617, -0.90536, 0.05267, -0.32536, -0.71021],
  [-0.05093, -0.06550, 0.44470, 0.10511, 0.67751, 0.72053, 0.10376, 0.21732, 0.07119, -0.96703, -0.09393, 0.17193, -0.73741, -0.06504, 0.32477, -0.91245, 0.06101, -0.30377, -0.73499],
  [-0.05293, -0.06640, 0.44578, 0.09388, 0.67684, 0.72395, 0.09476, 0.15536, 0.05792, -1.00485, -0.10214, 0.16456, -0.72823, -0.08559, 0.33180, -0.91824, 0.07641, -0.27847, -0.75957],
  [-0.05519, -0.06753, 0.44767, 0.08371, 0.67652, 0.72648, 0.08679, 0.09459, 0.02908, -1.01695, -0.10797, 0.15397, -0.71810, -0.09974, 0.33731, -0.91660, 0.09359, -0.24760, -0.78023],
  [-0.05737, -0.06786, 0.45013, 0.07555, 0.67643, 0.72835, 0.07902, 0.04112, -0.00867, -1.00484, -0.11713, 0.14079, -0.70063, -0.10679, 0.33580, -0.90562, 0.10379, -0.20611, -0.80257],
  [-0.05902, -0.06750, 0.45259, 0.06955, 0.67639, 0.72974, 0.07166, -0.00802, -0.04812, -0.97086, -0.12849, 0.12875, -0.68162, -0.11000, 0.33418, -0.89653, 0.11106, -0.15600, -0.82436],
  [-0.06014, -0.06728, 0.45473, 0.06529, 0.67706, 0.73017, 0.06466, -0.04634, -0.08522, -0.91782, -0.13812, 0.11688, -0.66726, -0.11211, 0.33047, -0.88803, 0.12050, -0.11094, -0.82888],
  [-0.06076, -0.06699, 0.45632, 0.06194, 0.67903, 0.72922, 0.05766, -0.06528, -0.11733, -0.84635, -0.14799, 0.10563, -0.66086, -0.11463, 0.32168, -0.87639, 0.12842, -0.08373, -0.81427],
  [-0.06076, -0.06581, 0.45689, 0.05907, 0.68211, 0.72715, 0.04980, -0.09067, -0.13459, -0.79078, -0.16026, 0.09876, -0.66390, -0.11722, 0.31041, -0.86469, 0.12938, -0.07522, -0.79244],
  [-0.06002, -0.06342, 0.45668, 0.05695, 0.68646, 0.72384, 0.03970, -0.11310, -0.13507, -0.75190, -0.17638, 0.09583, -0.67341, -0.11685, 0.29881, -0.85512, 0.11951, -0.08701, -0.76523],
  [-0.05895, -0.06019, 0.45591, 0.05452, 0.69199, 0.71930, 0.02798, -0.10891, -0.11880, -0.76040, -0.19524, 0.09555, -0.68733, -0.11594, 0.28492, -0.84398, 0.10343, -0.11727, -0.73485],
  [-0.05808, -0.05613, 0.45477, 0.05094, 0.69801, 0.71410, 0.01549, -0.07747, -0.10810, -0.79850, -0.21422, 0.10028, -0.70326, -0.11680, 0.26761, -0.82950, 0.08650, -0.15288, -0.70526],
  [-0.05753, -0.05231, 0.45338, 0.04639, 0.70439, 0.70828, 0.00319, -0.03301, -0.11085, -0.82038, -0.23428, 0.10488, -0.72150, -0.11899, 0.24809, -0.81580, 0.07317, -0.18467, -0.68128],
  [-0.05747, -0.04923, 0.45198, 0.04068, 0.71071, 0.70225, -0.00904, -0.01758, -0.10790, -0.84636, -0.25605, 0.10873, -0.74243, -0.12189, 0.22486, -0.79935, 0.06314, -0.21215, -0.65875],
  [-0.05792, -0.04647, 0.45109, 0.03399, 0.71593, 0.69703, -0.02095, 0.00006, -0.10366, -0.86946, -0.27510, 0.11554, -0.76366, -0.12523, 0.20531, -0.78451, 0.05646, -0.23556, -0.63572],
  [-0.05882, -0.04361, 0.45019, 0.02713, 0.71949, 0.69326, -0.03141, 0.03287, -0.09512, -0.88789, -0.29004, 0.12726, -0.79395, -0.12931, 0.19555, -0.77718, 0.05211, -0.25003, -0.62144],
  [-0.05992, -0.04054, 0.44873, 0.02067, 0.72158, 0.69085, -0.04025, 0.05403, -0.08453, -0.90197, -0.29695, 0.13986, -0.83528, -0.13466, 0.19422, -0.77640, 0.04861, -0.25566, -0.61785],
  [-0.06108, -0.03739, 0.44707, 0.01493, 0.72265, 0.68946, -0.04700, 0.06789, -0.07218, -0.91362, -0.29616, 0.14736, -0.89468, -0.14136, 0.19899, -0.78058, 0.04652, -0.25549, -0.62093],
  [-0.06250, -0.03430, 0.44976, 0.00997, 0.72297, 0.68887, -0.05169, 0.08011, -0.06003, -0.91923, -0.27336, 0.15206, -0.96575, -0.14895, 0.20538, -0.78397, 0.04516, -0.25292, -0.62641]
]

SPIN_LEN = len(spin)
JOINT_INDEX_START = 7
JOINT_NUMS = 12
TIMESTEP = 1 / 30
NEXT_INDEX = 1
DISTANCEMENT_THRESHOLD = 2
CONSTANT_FACTOR = 2000 


def calculate_ring_velocity(ma_array):
    ma_next = np.vstack((ma_array[NEXT_INDEX:, :], ma_array[:NEXT_INDEX, :]))
    ma_displacement = ma_next - ma_array
    ma_v = ma_displacement / TIMESTEP
    ma_v_norm = np.linalg.norm(ma_v, axis=1, keepdims=True)
    ma_weight = ma_v_norm / np.max(ma_v_norm)
    return ma_v, ma_v_norm, ma_weight
    

def sigmoid(x: np.array):
    # return .5 * x
    return (1 / (1 + np.exp(-x)) - .5)



def sample_random_point(ma_array):
    """
    当前维度的范围是[a,b],则生成的随机数范围是[a-(b-a),b+(b-a)],维度大小=JOINT_NUMS
    """
    point = []
    for i in range(JOINT_NUMS):
        dim_min, dim_max = min(ma_array[:, i]), max(ma_array[:, i])
        point.append(np.random.uniform(
            2 * dim_min - dim_max, 2 * dim_max - dim_min))
    return point


def sample_random_point_pi():
    """
    生成的随机数范围是[-pi,pi],维度大小=JOINT_NUMS
    """
    point = np.random.uniform(-np.pi, np.pi, size=JOINT_NUMS)
    return point


def calculate_point_normal_direction(ma_array, point):
    ma_v, ma_v_norm, ma_weight, point2ring_displacement, point2ring_nearest_displacement, point2ring_displacement_norm, point2ring_nearest_index, ring_nearest_index_v, ring_nearest_index_v_norm, distances_flag = new_func(ma_array, point)
    if distances_flag:
        displacement_on_ring_projection = np.dot(point2ring_nearest_displacement, ring_nearest_index_v) / ring_nearest_index_v_norm
        point2ring_normal_vector = point2ring_nearest_displacement - displacement_on_ring_projection * ring_nearest_index_v / ring_nearest_index_v_norm
        theta = np.dot(point2ring_normal_vector, ring_nearest_index_v)
        assert np.abs(theta) < 0.01, f'{theta} >= 0.01, the normal vector is not perpendicular to ring velocity'
        normal_direction = point2ring_normal_vector / np.linalg.norm(point2ring_normal_vector)
    else:
        forces = ma_weight * point2ring_displacement / (point2ring_displacement_norm ** 2)
        force = np.sum(forces, axis=0) / forces.shape[0]
        normal_direction = force / np.linalg.norm(force)
    return normal_direction

def new_func(ma_array, point):
    ma_v, ma_v_norm, ma_weight = calculate_ring_velocity(ma_array)
    point2ring_displacement = ma_array - point
    point2ring_displacement_norm = np.linalg.norm(
        point2ring_displacement, axis=1, keepdims=True)
    # x^(1/6)
    # point2ring_displacement_norm = point2ring_displacement_norm**(1/6)
    point2ring_nearest_index_temp = np.argmin(point2ring_displacement_norm)
    # if angle is greater than 90°, index + 1
    if np.dot(point2ring_displacement[point2ring_nearest_index_temp],
              ma_v[point2ring_nearest_index_temp]) < 0:
        point2ring_nearest_index = point2ring_nearest_index_temp + 1
        point2ring_nearest_index %= SPIN_LEN
    else:
        point2ring_nearest_index = point2ring_nearest_index_temp
    point2ring_nearest_displacement = point2ring_displacement[point2ring_nearest_index]
    ring_nearest_index_v = ma_v[point2ring_nearest_index]
    ring_nearest_index_v_norm = np.linalg.norm(ring_nearest_index_v)
    distances_flag = point2ring_displacement_norm[point2ring_nearest_index] < ma_v_norm[
        point2ring_nearest_index] * TIMESTEP * DISTANCEMENT_THRESHOLD
        
    return ma_v, ma_v_norm,ma_weight,point2ring_displacement,point2ring_nearest_displacement,point2ring_displacement_norm,point2ring_nearest_index,ring_nearest_index_v,ring_nearest_index_v_norm,distances_flag


def calculate_point_displacement(ma_array, point, displacement):
    ma_v, ma_v_norm,ma_weight,point2ring_displacement,point2ring_nearest_displacement,point2ring_displacement_norm,point2ring_nearest_index,ring_nearest_index_v,ring_nearest_index_v_norm,distances_flag = new_func(ma_array, point)
    displacement_normalize = displacement / np.linalg.norm(displacement)
    return displacement_normalize * ring_nearest_index_v_norm

def calculate_point_tangent_velocity(ma_array, point, decay=10):  # 200
    ma_v, ma_v_norm, ma_weight, point2ring_displacement, point2ring_nearest_displacement, point2ring_displacement_norm, point2ring_nearest_index, ring_nearest_index_v, ring_nearest_index_v_norm, distances_flag = new_func(ma_array, point)
    
    if point2ring_displacement_norm[point2ring_nearest_index] < 1e-4:
        return ma_v[point2ring_nearest_index]

    forces = ma_v / (point2ring_displacement_norm ** 2) / (decay * point2ring_displacement_norm + 1)
    force = np.sum(forces, axis=0)
    distance = np.sum(1 / (point2ring_displacement_norm ** 2), axis=0)
    velocity = force / distance
    # if np.linalg.norm(point2ring_nearest_displacement) < ma_v_norm[point2ring_nearest_index] * TIMESTEP * DISTANCEMENT_THRESHOLD:
    if distances_flag:
        tangent_velocity = (np.dot(velocity, ring_nearest_index_v) * ring_nearest_index_v / ring_nearest_index_v_norm ** 2)
    else:
        tangent_velocity = velocity
    return tangent_velocity


def repulse(normal_direction, ma_array, point):
    ma_v, ma_v_norm,ma_weight,point2ring_displacement,point2ring_nearest_displacement,point2ring_displacement_norm,point2ring_nearest_index,ring_nearest_index_v,ring_nearest_index_v_norm,distances_flag = new_func(ma_array, point)
    speed = CONSTANT_FACTOR / np.sum(1 / point2ring_displacement_norm * ma_weight) / SPIN_LEN
    normal_displacement = normal_direction * speed
    return normal_displacement




if __name__ == '__main__':
    sma = np.array(spin)[:, JOINT_INDEX_START: JOINT_INDEX_START + JOINT_NUMS]

    SAMPLE_POINT_NUMS = int(1)
    ITER_TIMES = 10
    input_list = []
    output_list = []

    for _ in tqdm.tqdm(range(SAMPLE_POINT_NUMS // 2)):
        point = sample_random_point(sma)
        for i in range(ITER_TIMES):
            input_list.append(copy.deepcopy(point))
            normal_direction = calculate_point_normal_direction(sma, point)
            normal_displacement = repulse(normal_direction, sma, point)
            tangent_displacement = calculate_point_tangent_velocity(sma, point)
            displacement =  tangent_displacement + normal_displacement 
            displacement = calculate_point_displacement(sma, point, displacement)
            point += displacement * TIMESTEP
            output_list.append(displacement)
        
    for _ in tqdm.tqdm(range(int(SAMPLE_POINT_NUMS // 2))):
        point = sample_random_point_pi()
        for i in range(ITER_TIMES):
            input_list.append(copy.deepcopy(point))
            normal_direction = calculate_point_normal_direction(sma, point)
            normal_displacement = repulse(normal_direction, sma, point)
            tangent_displacement = calculate_point_tangent_velocity(sma, point)
            displacement =  tangent_displacement + normal_displacement 
            displacement = calculate_point_displacement(sma, point, displacement)
            point += displacement * TIMESTEP
            output_list.append(displacement)

    sma_v, _, _ = calculate_ring_velocity(sma)
    input_list.append(sma)
    output_list.append(sma_v)
    input_array = np.vstack(input_list)
    output_array = np.vstack(output_list)
       
    allresult = {'input': input_array, 'output': output_array, 'ITER_TIMES': ITER_TIMES, 'SAMPLE_POINT_NUMS': SAMPLE_POINT_NUMS}
    file_path = f'collect_dataset/spin_data_V_{NOWTIME}_{SAMPLE_POINT_NUMS}_{ITER_TIMES}.pkl'
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(file_path, 'wb') as f:
        pickle.dump(allresult, f)