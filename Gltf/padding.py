import os
import numpy as np

def add_center_padding(array, target_shape=(300, 300)):
    y_diff = target_shape[0] - array.shape[0]
    x_diff = target_shape[1] - array.shape[1]

    y_pad_top = y_diff // 2
    y_pad_bottom = y_diff - y_pad_top

    x_pad_left = x_diff // 2
    x_pad_right = x_diff - x_pad_left

    return np.pad(array, ((y_pad_top, y_pad_bottom), (x_pad_left, x_pad_right)), 'constant', constant_values=0)

if __name__ == "__main__":
    input_folder = "dataset_numpy_train"
    output_folder = "dataset_numpy_train_padded"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for person_folder in sorted(os.listdir(input_folder)):
        person_path = os.path.join(input_folder, person_folder)
        padded_person_path = os.path.join(output_folder, person_folder)

        if not os.path.exists(padded_person_path):
            os.makedirs(padded_person_path)

        if os.path.isdir(person_path):
            for npy_file in sorted(os.listdir(person_path)):
                if npy_file.endswith('.npy'):
                    file_path = os.path.join(person_path, npy_file)
                    array = np.load(file_path)
                    
                    padded_array = add_center_padding(array)
                    
                    output_path = os.path.join(padded_person_path, npy_file)
                    np.save(output_path, padded_array)
                    print(f"Saved padded array to {output_path}")
