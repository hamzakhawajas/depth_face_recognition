import os
import numpy as np
from scipy.ndimage import zoom

def downsample_arrays(input_folder, output_folder, target_shape):
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.npy'):
                
                # Extract the person and file numbers from the file name
                person_number, file_number = map(int, file.split('_')[:2])

                depth_data = np.load(os.path.join(root, file))
                depth_data[np.isnan(depth_data)] = 0
                depth_data[(depth_data > 1.4)] = 0
                # if non_zero_indices[0].size > 0:
                #     min_val = np.min(depth_image[non_zero_indices])  
                #     max_val = np.max(depth_image)  
                    
                #     # Normalize the values to bring out features
                #     depth_image[non_zero_indices] = (depth_image[non_zero_indices] - min_val) / (max_val - min_val) 

                # Calculate the factors needed for downsampling
                y_factor = target_shape[0] / depth_data.shape[0]
                x_factor = target_shape[1] / depth_data.shape[1]

                # Downsample the depth data to the target shape
                downsampled_depth_data = zoom(depth_data, (y_factor, x_factor))

                # Create the output directory based on person number if it doesn't exist
                person_output_folder = os.path.join(output_folder, f'{person_number:03d}')
                os.makedirs(person_output_folder, exist_ok=True)

                output_file = os.path.join(person_output_folder, file)
                np.save(output_file, downsampled_depth_data)

def main():
    input_folder = 'final_depth_numpy'
    output_folder = 'downsampled_depth_numpy'
    target_shape = (54, 53)  # This is the minimum dimension you mentioned

    downsample_arrays(input_folder, output_folder, target_shape)

if __name__ == "__main__":
    main()
