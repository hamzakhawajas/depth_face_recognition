import os
import numpy as np

def find_max_indices(folder_path):
    max_x_index = max_y_index = -1

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.npy'):
                depth_data = np.load(os.path.join(root, file))
                cur_max_x_index = depth_data.shape[1] - 1
                cur_max_y_index = depth_data.shape[0] - 1
                max_x_index = max(max_x_index, cur_max_x_index)
                max_y_index = max(max_y_index, cur_max_y_index)

    return max_x_index, max_y_index

def pad_depth_arrays(input_folder, output_folder, max_x_index, max_y_index):
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.npy'):

                person_number, file_number = map(int, file.split('_')[:2])

                depth_data = np.load(os.path.join(root, file))
                depth_data[np.isnan(depth_data)] = 0  # Replace NaN with 0
                depth_data[depth_data > 1.3] = 0
                
                target_shape = (158, 155)
                
                
                # Calculate the padding for both dimensions
                pad_y = target_shape[0] - depth_data.shape[0]
                pad_x = target_shape[1] - depth_data.shape[1]

                # Divide the padding approximately equally on both sides
                pad_y_before, pad_y_after = pad_y // 2, pad_y - (pad_y // 2)
                pad_x_before, pad_x_after = pad_x // 2, pad_x - (pad_x // 2)

                # Pad the depth data to the target shape
                padded_depth_data = np.pad(
                    depth_data,
                    ((pad_y_before, pad_y_after), (pad_x_before, pad_x_after)),
                    mode='constant', constant_values=0  # Use 0 as the padding value
                )

                # Create the output directory based on person number if it doesn't exist
                person_output_folder = os.path.join(output_folder, f'{person_number:03d}')
                os.makedirs(person_output_folder, exist_ok=True)

                output_file = os.path.join(person_output_folder, file)
                np.save(output_file, padded_depth_data)


def main():
    input_folder = 'test_data/test1_depth_numpy'
    output_folder = 'test1_depth_numpy_center'

    max_x_index, max_y_index = find_max_indices(input_folder)
    pad_depth_arrays(input_folder, output_folder, max_x_index, max_y_index)

if __name__ == "__main__":
    main()
