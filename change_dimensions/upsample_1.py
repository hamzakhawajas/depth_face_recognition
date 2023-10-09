from scipy.ndimage import zoom
import os
import numpy as np
import cv2


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


from scipy.interpolate import griddata

def custom_interpolation(depth_data, target_shape):
    grid_x, grid_y = np.mgrid[0:target_shape[0]:1, 0:target_shape[1]:1]
    points = [(i, j) for i in range(depth_data.shape[0]) for j in range(depth_data.shape[1])]
    values = depth_data.flatten()
    return griddata(points, values, (grid_x, grid_y), method='cubic')

def resize_depth_arrays(input_folder, output_folder, max_x_index, max_y_index):
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.npy'):
                
                # Extract the person and file numbers from the file name
                person_number, file_number = map(int, file.split('_')[:2])

                depth_data = np.load(os.path.join(root, file))
                depth_data[np.isnan(depth_data)] = 10
                # target_shape = (max_y_index + 1, max_x_index + 1)
                target_shape = (158, 155)

                # Calculate the zoom factors
                y_zoom = target_shape[0] / depth_data.shape[0]
                x_zoom = target_shape[1] / depth_data.shape[1]

                # Perform the interpolation to resize
                # resized_depth_data = zoom(depth_data, (y_zoom, x_zoom), order=3)  # cubic interpolation
                #resized_depth_data = cv2.resize(depth_data, (155, 158), interpolation=cv2.INTER_LANCZOS4)
                resized_depth_data = custom_interpolation(depth_data, (158, 155))

                # Create the output directory based on person number if it doesn't exist
                person_output_folder = os.path.join(output_folder, f'{person_number:03d}')
                os.makedirs(person_output_folder, exist_ok=True)

                output_file = os.path.join(person_output_folder, file)
                np.save(output_file, resized_depth_data)

def main():
    input_folder = 'final_depth_numpy'
    output_folder = 'final_depth_numpy_inter'

    max_x_index, max_y_index = find_max_indices(input_folder)
    resize_depth_arrays(input_folder, output_folder, max_x_index, max_y_index)

if __name__ == "__main__":
    main()
