import os
import numpy as np
import matplotlib.pyplot as plt

def read_txt_to_depth_image(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()[1:]
        depth_image = np.loadtxt(lines)
    return depth_image

def apply_transformations(depth_image):
    depth_image = np.nan_to_num(depth_image, nan=0.0) 
    depth_image[depth_image > 1.3] = 0 
    
    non_zero_indices = np.nonzero(depth_image)
    if non_zero_indices[0].size > 0:
        min_val = np.min(depth_image[non_zero_indices])
        max_val = np.max(depth_image)
        
        depth_image[non_zero_indices] = (depth_image[non_zero_indices] - min_val) / (max_val - min_val)
    
    depth_image = np.power(depth_image, 1.5)
    depth_image = 1 - depth_image

    return depth_image

def main():
    depth_folder = 'test1_result'
    depth_files = [f for f in os.listdir(depth_folder) if f.endswith('_face.txt')]
    #print(depth_files)

    for depth_file in depth_files:
        person, file_number_str = depth_file.split('_')[:2]
        file_number = int(file_number_str)
        person = int(person)

        print(f"person: {person}  image: {file_number}")

        input_text_path = os.path.join(depth_folder, depth_file)
        print(input_text_path)
        final_depth_folder = f'test1_result/'
        
        if not os.path.exists(final_depth_folder):
            os.makedirs(final_depth_folder)

        if not os.path.exists(input_text_path):
            continue

        depth_image = read_txt_to_depth_image(input_text_path)
        transformed_depth_image = apply_transformations(depth_image)
        
        output_image_path = f"{final_depth_folder}/{str(person).zfill(3)}_{str(file_number).zfill(2)}_face.png"
        plt.imsave(output_image_path, transformed_depth_image, cmap='gray')

if __name__ == "__main__":
    main()
