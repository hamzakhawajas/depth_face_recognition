import os
import numpy as np

def find_max_indices(folder_path):
    max_x_index = max_y_index = -1

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.npy'):
                depth_data = np.load(os.path.join(root, file))
                cur_max_x_index = depth_data.shape[1] - 1
                cur_max_y_index = depth_data.shape[0] - 1
                max_x_index = max(max_x_index, cur_max_x_index)
                max_y_index = max(max_y_index, cur_max_y_index)

    return max_x_index, max_y_index

def pad_depth_arrays(input_folder, output_folder, max_x_index, max_y_index):
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.npy'):
                depth_data = np.load(os.path.join(root, file))
                depth_data[np.isnan(depth_data)] = 10
                target_shape = (158, 155)

                padded_depth_data = np.pad(
                    depth_data,
                    ((0, target_shape[0] - depth_data.shape[0]), (0, target_shape[1] - depth_data.shape[1])),
                    mode='constant', constant_values=10
                )

                rel_path = os.path.relpath(root, input_folder)
                output_subfolder = os.path.join(output_folder, rel_path)
                
                try:
                    os.makedirs(output_subfolder, exist_ok=True)
                    print(f"Successfully created the directory {output_subfolder}")
                except Exception as e:
                    print(f"Failed to create directory. Error: {e}")

                output_file = os.path.join(output_subfolder, file)
                np.save(output_file, padded_depth_data)

def main():
    input_folder = 'test_data/test1_depth_numpy'
    output_folder = 'test_data/test1_padded_depth_numpy'
    max_x_index, max_y_index = find_max_indices(input_folder)
    pad_depth_arrays(input_folder, output_folder, max_x_index, max_y_index)

if __name__ == "__main__":
    main()

