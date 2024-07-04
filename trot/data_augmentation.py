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
import numpy as np
import os
import tqdm



NOWTIME = time.strftime("%m_%d", time.localtime())
tort = [
  [0.00000, 0.00000, 0.41758, 0.48548, 0.51475, 0.52474, 0.47328, -0.13469, 0.19749, -0.98728, -0.29301, -0.20471, -1.23984, -0.23355, 0.42011, -1.21791, -0.18938, 0.26441, -0.94834],
  [0.03200, 0.00408, 0.42029, 0.48243, 0.51907, 0.51920, 0.47777, -0.14512, 0.26054, -0.95961, -0.29102, -0.36855, -1.18856, -0.27333, 0.31933, -1.19782, -0.16195, 0.32940, -0.92682],
  [0.06473, 0.00874, 0.42564, 0.47905, 0.52524, 0.51225, 0.48191, -0.15306, 0.30261, -0.90729, -0.25868, -0.48532, -1.10858, -0.30688, 0.20205, -1.13190, -0.13235, 0.37537, -0.87545],
  [0.09805, 0.01381, 0.43253, 0.47488, 0.53203, 0.50551, 0.48568, -0.16337, 0.32781, -0.83305, -0.22920, -0.55950, -1.03486, -0.33973, 0.07726, -1.03591, -0.10455, 0.40886, -0.80869],
  [0.13131, 0.01910, 0.44028, 0.46993, 0.53831, 0.49986, 0.48942, -0.16965, 0.34171, -0.74659, -0.23311, -0.60282, -0.94829, -0.36301, -0.03726, -0.92726, -0.07940, 0.43395, -0.73493],
  [0.16397, 0.02376, 0.44753, 0.46518, 0.54270, 0.49454, 0.49448, -0.17625, 0.35017, -0.65978, -0.19650, -0.61928, -0.86229, -0.37492, -0.13853, -0.82065, -0.06165, 0.46502, -0.67749],
  [0.19598, 0.02731, 0.45305, 0.46190, 0.54454, 0.48996, 0.50006, -0.17989, 0.35836, -0.58379, -0.11653, -0.58522, -0.80736, -0.37301, -0.22084, -0.73131, -0.05417, 0.51236, -0.66087],
  [0.22701, 0.02987, 0.45625, 0.45985, 0.54467, 0.48774, 0.50397, -0.18044, 0.36833, -0.54932, -0.08977, -0.53380, -0.76223, -0.36188, -0.28908, -0.66566, -0.05688, 0.56185, -0.67457],
  [0.25694, 0.03160, 0.45624, 0.45837, 0.54399, 0.48857, 0.50524, -0.17787, 0.39126, -0.61386, -0.09028, -0.47221, -0.71330, -0.33558, -0.32496, -0.62603, -0.06434, 0.61170, -0.71491],
  [0.28671, 0.03236, 0.45280, 0.45725, 0.54283, 0.49077, 0.50537, -0.16933, 0.42446, -0.72625, -0.08722, -0.39813, -0.70583, -0.30456, -0.33030, -0.61354, -0.07068, 0.66141, -0.78657],
  [0.31687, 0.03268, 0.44655, 0.45741, 0.54189, 0.49284, 0.50423, -0.17410, 0.44456, -0.86169, -0.08785, -0.30924, -0.75960, -0.27382, -0.31084, -0.62046, -0.07733, 0.69563, -0.87640],
  [0.34689, 0.03361, 0.43884, 0.45832, 0.54020, 0.49555, 0.50255, -0.18090, 0.44405, -1.01111, -0.09562, -0.21583, -0.83420, -0.26601, -0.24525, -0.67550, -0.08763, 0.71085, -0.97991],
  [0.37605, 0.03510, 0.43093, 0.45954, 0.53727, 0.49932, 0.50084, -0.19220, 0.40517, -1.15108, -0.10542, -0.11626, -0.90293, -0.27122, -0.14297, -0.76737, -0.10005, 0.70747, -1.08610],
  [0.40496, 0.03641, 0.42368, 0.46168, 0.53356, 0.50286, 0.49930, -0.20451, 0.30068, -1.25100, -0.12191, -0.01085, -0.96244, -0.27744, -0.03505, -0.84941, -0.11237, 0.67923, -1.17895],
  [0.43398, 0.03672, 0.41729, 0.46529, 0.52872, 0.50534, 0.49859, -0.21398, 0.14288, -1.31734, -0.13083, 0.09313, -1.00631, -0.27792, 0.05942, -0.90905, -0.12975, 0.62727, -1.25950],
  [0.46252, 0.03630, 0.41416, 0.46857, 0.52247, 0.50815, 0.49925, -0.20933, -0.03553, -1.31434, -0.13410, 0.18246, -1.01992, -0.26713, 0.15417, -0.95626, -0.15533, 0.55131, -1.31573],
  [0.49003, 0.03516, 0.41387, 0.47390, 0.51745, 0.50934, 0.49823, -0.18514, -0.20865, -1.24941, -0.13137, 0.24918, -1.00930, -0.24675, 0.23442, -0.97250, -0.18472, 0.44756, -1.32951],
  [0.51703, 0.03295, 0.41571, 0.48140, 0.51364, 0.50948, 0.49483, -0.15200, -0.35830, -1.14094, -0.12812, 0.29595, -0.98447, -0.22266, 0.30016, -0.96332, -0.21647, 0.32512, -1.29471],
  [0.54507, 0.02911, 0.42020, 0.48926, 0.50925, 0.51065, 0.49044, -0.10030, -0.47457, -0.99850, -0.12894, 0.32647, -0.93602, -0.19455, 0.35509, -0.93407, -0.24689, 0.19552, -1.21388],
  [0.57475, 0.02483, 0.42619, 0.49668, 0.50559, 0.51401, 0.48319, -0.07595, -0.54361, -0.90392, -0.13124, 0.34496, -0.86624, -0.16334, 0.39909, -0.89050, -0.27480, 0.06162, -1.09754],
  [0.60593, 0.02022, 0.43297, 0.50384, 0.50198, 0.51823, 0.47496, -0.07796, -0.59147, -0.83841, -0.13472, 0.35335, -0.78171, -0.13330, 0.43503, -0.83421, -0.29635, -0.06430, -0.96968],
  [0.63802, 0.01588, 0.43983, 0.51004, 0.49891, 0.52258, 0.46677, -0.02444, -0.57695, -0.76569, -0.14084, 0.35431, -0.68742, -0.11056, 0.45942, -0.76450, -0.31035, -0.17816, -0.84233],
  [0.66995, 0.01220, 0.44597, 0.51492, 0.49699, 0.52599, 0.45955, 0.03306, -0.51857, -0.74178, -0.14370, 0.34841, -0.59691, -0.09653, 0.48009, -0.70593, -0.31628, -0.27009, -0.73351],
  [0.70088, 0.00959, 0.44993, 0.51822, 0.49621, 0.52766, 0.45475, 0.03693, -0.45715, -0.71912, -0.14406, 0.33574, -0.54469, -0.09372, 0.50782, -0.68224, -0.31207, -0.34200, -0.66205],
  [0.73035, 0.00804, 0.45044, 0.52031, 0.49605, 0.52720, 0.45307, 0.02213, -0.38793, -0.73939, -0.14644, 0.34990, -0.58431, -0.10378, 0.53688, -0.68518, -0.29773, -0.36876, -0.64115],
  [0.75870, 0.00730, 0.44774, 0.52114, 0.49565, 0.52560, 0.45440, -0.00440, -0.30834, -0.79874, -0.15446, 0.38900, -0.69914, -0.12304, 0.56939, -0.71815, -0.27581, -0.36458, -0.64736],
  [0.78645, 0.00701, 0.44285, 0.52041, 0.49576, 0.52325, 0.45783, -0.01870, -0.22990, -0.84639, -0.17108, 0.43025, -0.84662, -0.14480, 0.60458, -0.77773, -0.24323, -0.34274, -0.65512],
  [0.81350, 0.00700, 0.43717, 0.51825, 0.49603, 0.52142, 0.46204, -0.03190, -0.13705, -0.89790, -0.19256, 0.44180, -1.00410, -0.16929, 0.62965, -0.84436, -0.21962, -0.29603, -0.68556],
  [0.83943, 0.00731, 0.43142, 0.51414, 0.49568, 0.52106, 0.46739, -0.05247, -0.04078, -0.94286, -0.22764, 0.38617, -1.12246, -0.19891, 0.64651, -0.92152, -0.21120, -0.20773, -0.75772],
  [0.86408, 0.00766, 0.42589, 0.50811, 0.49510, 0.52151, 0.47406, -0.06905, 0.04676, -0.96850, -0.26272, 0.27126, -1.20372, -0.23094, 0.65590, -1.00493, -0.20474, -0.09638, -0.84458],
  [0.88800, 0.00537, 0.42062, 0.50131, 0.49551, 0.52224, 0.48002, -0.08727, 0.13061, -0.98216, -0.30963, 0.11052, -1.24698, -0.26064, 0.64843, -1.08140, -0.19889, 0.00708, -0.92231],
  [0.91180, 0.00341, 0.41688, 0.49456, 0.49654, 0.52247, 0.48568, -0.09822, 0.20722, -0.98374, -0.34176, -0.06244, -1.25238, -0.28866, 0.62110, -1.14684, -0.18915, 0.09785, -0.97609],
  [0.93565, 0.00000, 0.41758, 0.48849, 0.49741, 0.52141, 0.49203, -0.10972, 0.27104, -0.96950, -0.35302, -0.23058, -1.21682, -0.31460, 0.57537, -1.20075, -0.16983, 0.18096, -1.01378]
]

TORT_LEN = len(tort)
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
        point2ring_nearest_index %= TORT_LEN
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
    speed = CONSTANT_FACTOR / np.sum(1 / point2ring_displacement_norm * ma_weight) / TORT_LEN
    normal_displacement = normal_direction * speed
    return normal_displacement




if __name__ == '__main__':
    tma = np.array(tort)[:, JOINT_INDEX_START: JOINT_INDEX_START + JOINT_NUMS]

    SAMPLE_POINT_NUMS = int(1e1)
    ITER_TIMES = 100
    input_list = []
    output_list = []

    for _ in tqdm.tqdm(range(SAMPLE_POINT_NUMS // 2)):
        point = sample_random_point(tma)
        for i in range(ITER_TIMES):
            input_list.append(copy.deepcopy(point))
            normal_direction = calculate_point_normal_direction(tma, point)
            normal_displacement = repulse(normal_direction, tma, point)
            tangent_displacement = calculate_point_tangent_velocity(tma, point)
            displacement =  tangent_displacement + normal_displacement 
            displacement = calculate_point_displacement(tma, point, displacement)
            point += displacement * TIMESTEP
            output_list.append(displacement)
        
    for _ in tqdm.tqdm(range(int(SAMPLE_POINT_NUMS // 2))):
        point = sample_random_point_pi()
        for i in range(ITER_TIMES):
            input_list.append(copy.deepcopy(point))
            normal_direction = calculate_point_normal_direction(tma, point)
            normal_displacement = repulse(normal_direction, tma, point)
            tangent_displacement = calculate_point_tangent_velocity(tma, point)
            displacement =  tangent_displacement + normal_displacement 
            displacement = calculate_point_displacement(tma, point, displacement)
            point += displacement * TIMESTEP
            output_list.append(displacement)

    tma_v, _, _ = calculate_ring_velocity(tma)
    input_list.append(tma)
    output_list.append(tma_v)
    input_array = np.vstack(input_list)
    output_array = np.vstack(output_list)
       
    allresult = {'input': input_array, 'output': output_array, 'ITER_TIMES': ITER_TIMES, 'SAMPLE_POINT_NUMS': SAMPLE_POINT_NUMS}
    file_path = f'trot/dataset/tort_data_V_{NOWTIME}_{SAMPLE_POINT_NUMS}_{ITER_TIMES}.pkl'
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(file_path, 'wb') as f:
        pickle.dump(allresult, f)