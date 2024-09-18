import data
import augmentation_data


if __name__ == "__main__":
    trot = data.trot_array
    ma_v, _, _ = augmentation_data.calculate_ring_velocity(trot)
    data.ploter(trot, ma_v, )