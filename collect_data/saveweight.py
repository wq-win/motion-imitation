import copy
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from collections import deque
# from numba import njit
import matplotlib.pyplot as plt

pace = [
  [0.00000, 0.00000, 0.43701, 0.49491, 0.53393, 0.49912, 0.46997, -0.12721, 0.07675, -0.95545, -0.25301, 0.18682, -1.14403, -0.19362, 0.14030, -0.77823, -0.09528, 0.05437, -0.97596],
  [0.01641, 0.00223, 0.43771, 0.48959, 0.53669, 0.50119, 0.47018, -0.12680, 0.11820, -0.94606, -0.28172, 0.03357, -1.16456, -0.20247, 0.17747, -0.77104, -0.09744, -0.05174, -0.93399],
  [0.03278, 0.00476, 0.43896, 0.48274, 0.53845, 0.50530, 0.47084, -0.12518, 0.15584, -0.92492, -0.30683, -0.11684, -1.15057, -0.21314, 0.22216, -0.76688, -0.10566, -0.14981, -0.88721],
  [0.04882, 0.00706, 0.44055, 0.47656, 0.53895, 0.50939, 0.47217, -0.12177, 0.19077, -0.89664, -0.31713, -0.25529, -1.10780, -0.22449, 0.26890, -0.75917, -0.11668, -0.23940, -0.83319],
  [0.06588, 0.00883, 0.44210, 0.47093, 0.53858, 0.51304, 0.47428, -0.12087, 0.22330, -0.86296, -0.31269, -0.35342, -1.05107, -0.23663, 0.31297, -0.74403, -0.12989, -0.31593, -0.77429],
  [0.08286, 0.01033, 0.44397, 0.46634, 0.53774, 0.51543, 0.47717, -0.12319, 0.25043, -0.82652, -0.29625, -0.41874, -0.97904, -0.24941, 0.35253, -0.72424, -0.13472, -0.37088, -0.71971],
  [0.09884, 0.01171, 0.44605, 0.46371, 0.53747, 0.51525, 0.48022, -0.12780, 0.26864, -0.79123, -0.27806, -0.47452, -0.91527, -0.26153, 0.38053, -0.69634, -0.13017, -0.40471, -0.67348],
  [0.11564, 0.01337, 0.44783, 0.46172, 0.53682, 0.51397, 0.48422, -0.13502, 0.28660, -0.75117, -0.28068, -0.51479, -0.83098, -0.27053, 0.40688, -0.66852, -0.11715, -0.41160, -0.63590],
  [0.13247, 0.01557, 0.44866, 0.46017, 0.53653, 0.51145, 0.48867, -0.14270, 0.30399, -0.71655, -0.21607, -0.52903, -0.77176, -0.27539, 0.43360, -0.64668, -0.10623, -0.37913, -0.63582],
  [0.14967, 0.01760, 0.44808, 0.45870, 0.53617, 0.50991, 0.49205, -0.14760, 0.32861, -0.68749, -0.16653, -0.50534, -0.73515, -0.27867, 0.47223, -0.65260, -0.10817, -0.30617, -0.68954],
  [0.16688, 0.01936, 0.44679, 0.45801, 0.53697, 0.50846, 0.49332, -0.14926, 0.35277, -0.66859, -0.14587, -0.46223, -0.71030, -0.27905, 0.52277, -0.69369, -0.10386, -0.25300, -0.72486],
  [0.18457, 0.02113, 0.44491, 0.45884, 0.53861, 0.50792, 0.49131, -0.14788, 0.37881, -0.65994, -0.12952, -0.41299, -0.68721, -0.26795, 0.55542, -0.74544, -0.10247, -0.19779, -0.76080],
  [0.20315, 0.02255, 0.44338, 0.46159, 0.54103, 0.50733, 0.48666, -0.14168, 0.40260, -0.65184, -0.16240, -0.34246, -0.76043, -0.24415, 0.54899, -0.78040, -0.10484, -0.15274, -0.78151],
  [0.22288, 0.02351, 0.44195, 0.46521, 0.54355, 0.50682, 0.48089, -0.13629, 0.42854, -0.65674, -0.16512, -0.29743, -0.79699, -0.22199, 0.52110, -0.82139, -0.10857, -0.11585, -0.79425],
  [0.24349, 0.02420, 0.44024, 0.46907, 0.54564, 0.50727, 0.47426, -0.12753, 0.44960, -0.68466, -0.17700, -0.23585, -0.84852, -0.21056, 0.47836, -0.87137, -0.11339, -0.07894, -0.80477],
  [0.26441, 0.02473, 0.43823, 0.47325, 0.54703, 0.50712, 0.46865, -0.11968, 0.45798, -0.76842, -0.19037, -0.17536, -0.89554, -0.20650, 0.44393, -0.92752, -0.11522, -0.04086, -0.81465],
  [0.28558, 0.02464, 0.43620, 0.47835, 0.54689, 0.50616, 0.46464, -0.12197, 0.43800, -0.86342, -0.19421, -0.11435, -0.93678, -0.19062, 0.40079, -0.97485, -0.11837, -0.00403, -0.82084],
  [0.30563, 0.02398, 0.43551, 0.48404, 0.54571, 0.50448, 0.46196, -0.12253, 0.41137, -0.95532, -0.19119, -0.06019, -0.96439, -0.17122, 0.33514, -1.00492, -0.11824, 0.02713, -0.81663],
  [0.32565, 0.02340, 0.43543, 0.48863, 0.54314, 0.50286, 0.46192, -0.12639, 0.37667, -1.06982, -0.18340, -0.00535, -0.97909, -0.16186, 0.24489, -1.00477, -0.11752, 0.06483, -0.81530],
  [0.34513, 0.02264, 0.43600, 0.49248, 0.53972, 0.50166, 0.46315, -0.13757, 0.29075, -1.15527, -0.17385, 0.04504, -0.97993, -0.16089, 0.14852, -0.98629, -0.11866, 0.10202, -0.80805],
  [0.36352, 0.02156, 0.43678, 0.49651, 0.53506, 0.50047, 0.46553, -0.14822, 0.15752, -1.19552, -0.16643, 0.09010, -0.97221, -0.16121, 0.04991, -0.95571, -0.12166, 0.14042, -0.79969],
  [0.38133, 0.01988, 0.43839, 0.50053, 0.52944, 0.49854, 0.46971, -0.15608, 0.00492, -1.19387, -0.15410, 0.13039, -0.95309, -0.16926, -0.05029, -0.90903, -0.12653, 0.17847, -0.78722],
  [0.39841, 0.01837, 0.44059, 0.50384, 0.52281, 0.49739, 0.47479, -0.15983, -0.13783, -1.15844, -0.14303, 0.16753, -0.92657, -0.18165, -0.14768, -0.85218, -0.13280, 0.21770, -0.77474],
  [0.41513, 0.01669, 0.44294, 0.50649, 0.51629, 0.49694, 0.47954, -0.15788, -0.25191, -1.09422, -0.13183, 0.19619, -0.88943, -0.19374, -0.23443, -0.78796, -0.14329, 0.25547, -0.75380],
  [0.43177, 0.01473, 0.44521, 0.50907, 0.51037, 0.49701, 0.48306, -0.15548, -0.33574, -1.01839, -0.12400, 0.22032, -0.84860, -0.20385, -0.30441, -0.72873, -0.15418, 0.28667, -0.72651],
  [0.44802, 0.01314, 0.44760, 0.51097, 0.50622, 0.49819, 0.48420, -0.15480, -0.40127, -0.95462, -0.12135, 0.23766, -0.80623, -0.20927, -0.35212, -0.67100, -0.16640, 0.31242, -0.69448],
  [0.46489, 0.01140, 0.45021, 0.51169, 0.50451, 0.49981, 0.48356, -0.14707, -0.45308, -0.87697, -0.11954, 0.25095, -0.75704, -0.21183, -0.38106, -0.61525, -0.17028, 0.33364, -0.66014],
  [0.48202, 0.00922, 0.45259, 0.51313, 0.50273, 0.50274, 0.48085, -0.08334, -0.47429, -0.78665, -0.12094, 0.26006, -0.70915, -0.19570, -0.37232, -0.58446, -0.17731, 0.34955, -0.62396],
  [0.49978, 0.00692, 0.45301, 0.51399, 0.49995, 0.50649, 0.47889, -0.03442, -0.44226, -0.72261, -0.12344, 0.27515, -0.66283, -0.19484, -0.31002, -0.61017, -0.18530, 0.38462, -0.62481],
  [0.51793, 0.00522, 0.45155, 0.51416, 0.49704, 0.50996, 0.47805, -0.03327, -0.39805, -0.70482, -0.12444, 0.29801, -0.62361, -0.20098, -0.25119, -0.65274, -0.19097, 0.44064, -0.67343],
  [0.53748, 0.00395, 0.44913, 0.51411, 0.49529, 0.51164, 0.47813, -0.07110, -0.32111, -0.76216, -0.12382, 0.32680, -0.59657, -0.19979, -0.19735, -0.68530, -0.18261, 0.49555, -0.73822],
  [0.55659, 0.00280, 0.44697, 0.51485, 0.49644, 0.50985, 0.47805, -0.07549, -0.26493, -0.80317, -0.11877, 0.35467, -0.59078, -0.20209, -0.14052, -0.72329, -0.16125, 0.51232, -0.79071],
  [0.57577, 0.00167, 0.44507, 0.51650, 0.49783, 0.50696, 0.47789, -0.07131, -0.21348, -0.82662, -0.11386, 0.37332, -0.60466, -0.19963, -0.08971, -0.74502, -0.14228, 0.49408, -0.83879],
  [0.59497, 0.00154, 0.44312, 0.51800, 0.50013, 0.50330, 0.47774, -0.08576, -0.15562, -0.87133, -0.11535, 0.37462, -0.65372, -0.20273, -0.04893, -0.76230, -0.12839, 0.46938, -0.90473],
  [0.61400, 0.00191, 0.44160, 0.51843, 0.50324, 0.49974, 0.47774, -0.09484, -0.10177, -0.90486, -0.12731, 0.36711, -0.75677, -0.20671, -0.01317, -0.77205, -0.11790, 0.43685, -0.97024],
  [0.63303, 0.00233, 0.44009, 0.51746, 0.50557, 0.49713, 0.47905, -0.10425, -0.04639, -0.92873, -0.14730, 0.35852, -0.86271, -0.21267, 0.02886, -0.78466, -0.10501, 0.39079, -1.01703],
  [0.65155, 0.00263, 0.43886, 0.51586, 0.50831, 0.49492, 0.48016, -0.11146, 0.00547, -0.94503, -0.18348, 0.32590, -0.98252, -0.21663, 0.07145, -0.79413, -0.09216, 0.31946, -1.04176],
  [0.67005, 0.00126, 0.43824, 0.51299, 0.51174, 0.49342, 0.48113, -0.12047, 0.05387, -0.95210, -0.21892, 0.23998, -1.07604, -0.22485, 0.10828, -0.79239, -0.08403, 0.22582, -1.04134],
  [0.68773, 0.00000, 0.43701, 0.50903, 0.51581, 0.49242, 0.48203, -0.12785, 0.09815, -0.95073, -0.26299, 0.10340, -1.12756, -0.23415, 0.13683, -0.78085, -0.07723, 0.11886, -1.01564]
]


# print(goal_joint_velocity)
# print(i.as_quat())

def generate_data_random(random_num, data, noise_factor):
    imu_euler, joint_angle = data
    sample_dim = 4 + joint_angle.shape[1]
    ref_num = imu_euler.shape[0]
    new_data = {"input": np.empty((ref_num*random_num, sample_dim)),
            "label": np.empty((ref_num*random_num, joint_angle.shape[1]))}
    for i in range(random_num):
        imu_euler_noised = imu_euler + np.random.normal(0, 2 * np.pi * noise_factor, size=imu_euler.shape)
        imu_euler_noised_R = R.from_euler("XYZ", imu_euler_noised)
        imu_euler_noised_quat = imu_euler_noised_R.as_quat()
        # print(imu_euler_noised_quat)
        # Obs_noised = Obs + np.random.normal(0,,size=Obs.shape)

        joint_angle_noised = joint_angle + np.random.normal(0, 2 * np.pi * noise_factor, size=joint_angle.shape)
        goal_joint_angle = np.vstack((joint_angle[1:, :], joint_angle[:1, :]))
        goal_joint_velocity = ((goal_joint_angle - joint_angle_noised) / timestep)  # j_a_n - g_j_a

        new_data["input"][i*ref_num:(i+1)*ref_num, :] = np.hstack((imu_euler_noised_quat, joint_angle_noised))
        new_data["label"][i*ref_num:(i+1)*ref_num, :] = goal_joint_velocity
    return new_data

def repulse(fixed_points, moving_points, constant_factor):
    points = np.empty_like(moving_points)
    point_displacements = np.empty_like(moving_points)
    point_velocitys = np.empty_like(moving_points)
    for i in range(moving_points.shape[0]):
        displacement = moving_points[i] - fixed_points
        distance = np.linalg.norm(displacement, axis=1, keepdims=1)
        forces =  displacement / (distance ** 2)
        force = np.sum(forces, axis=0) / fixed_points.shape[0]
        force_mag = np.linalg.norm(force)
        direction = force/force_mag

        speed = 1/ np.sum( 1/distance )/ fixed_points.shape[0] * constant_factor


        new_displacement = direction * speed

        new_moving_points =moving_points[i] + new_displacement
        points[i, :] = new_moving_points
        # point_displacements[i, :] = new_displacement
        # new_displacement_size = np.linalg.norm(new_displacement)
        # direction = new_displacement/new_displacement_size
        # velocity_rate =1./ np.linalg.norm(1./distance)
        # point_velocitys[i, :]  = direction * velocity_rate
        point_velocitys[i, :] = new_displacement
    return points, point_velocitys

def generate_data_repulse(data, initial_times, iteration_num, timestep=1 / 30):
    imu_euler, joint_angle = data
    sample_dim = 4 + joint_angle.shape[1]
    ref_num = imu_euler.shape[0]

    input_list = []
    label_list = []

    # goal_joint_angle0 = np.vstack((joint_angle[1:, :], joint_angle[:1, :]))
    # goal_joint_velocity_1 = ((goal_joint_angle0 - joint_angle) / timestep)
    # input_list.apend(np.hstack((imu_euler, joint_angle)))
    # label_list.apend(goal_joint_velocity_1)
    original_data = generate_data_random(initial_times, data, noise_factor=0.0)
    input_list.append(original_data["input"])
    label_list.append(original_data["label"])

    # for i in range(initial_times):
    #     imu_euler_noised = imu_euler + np.random.normal(0, 2 * np.pi * noise_factor, size=imu_euler.shape)
    #     imu_euler_noised_R = R.from_euler("XYZ", imu_euler_noised)
    #     imu_euler_noised_quat = imu_euler_noised_R.as_quat()
    #     # print(imu_euler_noised_quat)
    #     # Obs_noised = Obs + np.random.normal(0,,size=Obs.shape)
    #     joint_angle_noise = np.random.normal(0, 2 * np.pi * noise_factor, size=joint_angle.shape)
    #     joint_angle_noised = joint_angle + joint_angle_noise
    #     goal_joint_velocity_1 = ((goal_joint_angle0 - joint_angle_noised) / timestep)
    noised_data = generate_data_random(initial_times, data, noise_factor=0.01)
    # input_list.append(noised_data["input"])
    # label_list.append(noised_data["label"])

    joint_angle_noised = noised_data["input"][:, 4:]
    imu_euler_noised_quat = noised_data["input"][:, :4]

    repulsed_joint_angle = joint_angle_noised
    goal_joint_angle = np.vstack((joint_angle[2:, :], joint_angle[:2, :]))
    goal_joint_angle_tiled = np.tile(goal_joint_angle, (initial_times, 1))
    goal_joint_displacements = (goal_joint_angle_tiled - joint_angle_noised)

    for i in range(iteration_num):
        repulsed_joint_angle, point_displacements = repulse(joint_angle, repulsed_joint_angle, constant_factor=100)
        goal_joint_angle = np.vstack((joint_angle[1:, :], joint_angle[:1, :]))
        # repulsed_joint_angle = repulsed_joint_angle + goal_joint_displacements
        # point_displacements = point_displacements + goal_joint_displacements
        goal_joint_velocity = -point_displacements/timestep
        # goal_joint_velocity = 1./point_displacements
        input_list.append(np.hstack((imu_euler_noised_quat, repulsed_joint_angle)))
        label_list.append(goal_joint_velocity)

    new_data = {"input": np.vstack(input_list),
                "label": np.vstack(label_list)}
    return new_data

def learning_loop(KCtoMBONweight, KC, learning_rate, epochs, activate_indices):
    loss_list = [ 0 for _ in range(epochs)]
    # indices1 = np.arange(KCtoMBONweight.shape[0])[:, None]
    # indices2 = np.arange(KC.shape[0])[:, None]

    for i in range(int(epochs)):
        # loss = (goal_joint_velocity - (KCtoMBONweight[indices1, activate_indices] @ KC[indices2, activate_indices].T).T
        MBON_input = (KCtoMBONweight @ KC.T).T
        loss = (goal_joint_velocity - MBON_input)
        gradient = loss.T @ KC
        KCtoMBONweight += gradient * learning_rate
        # print(KC @ KCtoMBONweight)
        loss = loss ** 2
        loss_list[i] = loss.mean()
        learning_rate *= epsilon
    return KCtoMBONweight, loss, loss_list, learning_rate, gradient.max()

def trjactory_ploter(data):
    # new_data = {"input": np.vstack(input_list),
    #             "label": np.vstack(label_list)}
    ax = plt.figure().add_subplot(projection='3d')

    # Make the grid
    x = data["input"][:, 4]
    y = data["input"][:, 5]
    z = data["input"][:, 6]

    # Make the direction data for the arrows
    u = data["label"][:, 0]
    v = data["label"][:, 1]
    w = data["label"][:, 2]

    ax.quiver(x, y, z, u, v, w, normalize=False, length=0.04)

    plt.show()

if __name__ == "__main__":

    pace = np.array(pace)
    ref_len = pace.shape[0]
    imu_quat = pace[:, 3:7]
    imu = R.from_quat(imu_quat)
    a = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    r = R.from_matrix(a)
    # r1 = r_trans*r

    imu_transformed = r.as_matrix() @ imu.as_matrix()
    imu_R = R.from_matrix(imu_transformed)
    imu_euler = imu_R.as_euler('XYZ', degrees=False)
    joint_angle = pace[:, 7:]
    loss_list = []

    method = 'field' # 'random' #
    timestep = 0.01667
    timestep = 1 / 30
    noise_factor = 0.01
    random_times = 10

    initial_times = 1
    interation_num = 10

    learning_rate = 1e-5
    epsilon = 1 - 2e-4
    epochs = 20  # 迭代次数
    episode = 1000
    batch_size = 512

    PN_dims = 16
    KC_dims = 10000
    MBON_dims = 12
    weights_PN2KC_full = np.random.normal(0, 1, (KC_dims, PN_dims))
    PN_smaple_rate = 0.25
    num_dim_PN_sampled = int(PN_dims * PN_smaple_rate)
    KC_activation_rate = 0.05 # percentage of KC activatity level
    num_dim_KC_activated = int(KC_dims * KC_activation_rate)  # remian PNtoKCweight top 3*1000 weight

    weight_sorted_indices = np.argsort(weights_PN2KC_full, axis=1)
    top_weight_indices = weight_sorted_indices[:, -num_dim_PN_sampled:]
    weights_PN2KC = np.zeros_like(weights_PN2KC_full)
    weights_PN2KC[np.arange(KC_dims)[:, None], top_weight_indices] = 1/num_dim_PN_sampled
    KCtoMBONweight = np.random.normal(0, 1/num_dim_KC_activated, size=(MBON_dims, KC_dims))

    data = (imu_euler, joint_angle)
    #
    if method == 'random':
        data = generate_data_random(random_times, data, noise_factor)
    elif method == 'field':
        data = generate_data_repulse(data, initial_times, interation_num)
    for akey in data.keys():
        data[akey] = np.array(data[akey])
    # print(len(data["input"]))
    # print('data["input"]' + str(data["input"][0:39]))
    # print('data["label"]' + str(data["label"][0:39]))

    trjactory_ploter(data)
    # for i in tqdm(range(episode)):
    #     sample_index = np.random.choice(data["input"].shape[0], batch_size)
    #     PN = np.array(data["input"][sample_index])
    #     goal_joint_velocity = np.array(data["label"][sample_index])
    #     KC_input = (weights_PN2KC @ PN.T).T
    #     sorted_indices = np.argsort(KC_input)
    #     activate_indices = sorted_indices[:, -num_dim_KC_activated:, ]
    #     inactivate_indices = sorted_indices[:, :-num_dim_KC_activated]
    #     KC_input[np.arange(KC_input.shape[0])[:, None], inactivate_indices] = 0
    #     # KC = KC_input/np.sum(KC_input, axis=1, keepdims=1)
    #     KC = KC_input

    #     KCtoMBONweight, loss, loss_list_epoch, learning_rate, gradient_max = \
    #         learning_loop(KCtoMBONweight, KC, learning_rate, epochs, activate_indices)
    #     loss_list.extend(loss_list_epoch)
    #     # print(loss)
    #     # with np.printoptions(threshold=np.inf):
    #     #     print("KC_input:")
    #     #     print(KC_input[0])
    #     tqdm.write("loss: %f, learning_rate: %f, max_KC_input[0]: %f, gradient_max: %f"
    #           %(loss_list[-1], learning_rate, max(KC_input[0]), gradient_max))

    # plt.plot(range(len(loss_list)), loss_list)
    # plt.show()
    # allresult = {'weights_PN2KC': weights_PN2KC,
    #              'PN': pace[:, 3:],
    #              'num_dim_KC_activated': num_dim_KC_activated, 'KCtoMBONweight': KCtoMBONweight}
    # # 使用pickle保存数组到文件

    # with open('weight_dataV0_01.pkl', 'wb') as f:
    #     pickle.dump(allresult, f)
