import os
import numpy as np
import matplotlib.pyplot as plt

def read_txt_to_depth_image(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()[1:]  # Skip the header line
        depth_image = np.loadtxt(lines)
    return depth_image

def apply_transformations(depth_image):
    depth_image = np.nan_to_num(depth_image, nan=0.0)  # Replace NaNs with 0
    depth_image[depth_image > 1.3] = 0  # Thresholding
    
    non_zero_indices = np.nonzero(depth_image)
    if non_zero_indices[0].size > 0:
        min_val = np.min(depth_image[non_zero_indices])  # Ignore zeros
        max_val = np.max(depth_image)
        
        depth_image[non_zero_indices] = (depth_image[non_zero_indices] - min_val) / (max_val - min_val)  # Normalization
    
    depth_image = np.power(depth_image, 1.5)  # Gamma correction
    
    depth_image = 1 - depth_image  # Invert the depth image so that less depth appears lighter

    return depth_image

def main():
    for person in range(25, 26):  # Assuming persons are numbered from 0 to 24
        for file_number in range(0, 12):  # Assuming file numbers are from 0 to 11
            print(person, " ", file_number)
            # Input and output directories
            input_text_path = f'output_face_text/{str(person).zfill(3)}/{str(person).zfill(3)}_{str(file_number).zfill(2)}_face.txt'
            final_depth_folder = f'final_depth/{str(person).zfill(3)}'
            
            # Create output directory if it does not exist
            if not os.path.exists(final_depth_folder):
                os.makedirs(final_depth_folder)
            
            if not os.path.exists(input_text_path):
                continue

            # Read and transform the depth image
            depth_image = read_txt_to_depth_image(input_text_path)
            transformed_depth_image = apply_transformations(depth_image)
            
            # Save the transformed depth image
            output_image_path = f"{final_depth_folder}/{str(person).zfill(3)}_{str(file_number).zfill(2)}_final_depth.png"
            plt.imsave(output_image_path, transformed_depth_image, cmap='gray')

if __name__ == "__main__":
    main()
