from scipy.interpolate import RectBivariateSpline
import os
import numpy as np

def resize_depth_arrays(input_folder, output_folder):
    target_shape = (158, 155)  # The target dimensions
    
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.npy'):
                
                # Extract the person and file numbers from the file name
                person_number, file_number = map(int, file.split('_')[:2])

                depth_data = np.load(os.path.join(root, file))
                depth_data[np.isnan(depth_data)] = 0  # Placeholder for NaN
                
                y = np.linspace(0, 1, depth_data.shape[0])
                x = np.linspace(0, 1, depth_data.shape[1])
                
                interpolator = RectBivariateSpline(y, x, depth_data)
                
                y_new = np.linspace(0, 1, target_shape[0])
                x_new = np.linspace(0, 1, target_shape[1])
                
                resized_depth_data = interpolator(y_new, x_new)

                # Create the output directory based on person number if it doesn't exist
                person_output_folder = os.path.join(output_folder, f'{person_number:03d}')
                os.makedirs(person_output_folder, exist_ok=True)

                output_file = os.path.join(person_output_folder, file)
                np.save(output_file, resized_depth_data)

def main():
    input_folder = 'final_depth_numpy'
    output_folder = 'final_depth_numpy_inter'

    resize_depth_arrays(input_folder, output_folder)

if __name__ == "__main__":
    main()
