import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def read_txt_to_depth_image(file_path, width, height):
    depth_image = np.zeros((height, width))
    with open(file_path, 'r') as f:
        next(f)  # Skip header line
        for line in f:
            i, j, x, y, z, _, _ = map(float, line.strip().split('\t'))
            if 0 <= int(i) < width and 0 <= int(j) < height:
                depth_image[int(j), int(i)] = z
    return depth_image

def apply_transformations(depth_image):
    depth_image = np.nan_to_num(depth_image, nan=0.0) 
    depth_image[depth_image > 1.3] = 0 

    non_zero_indices = np.nonzero(depth_image)
    if non_zero_indices[0].size > 0:
        min_val = np.min(depth_image[non_zero_indices])  
        max_val = np.max(depth_image)
        depth_image[non_zero_indices] = (depth_image[non_zero_indices] - min_val) / (max_val - min_val)  # Normalization

    depth_image = np.power(depth_image, 1.5) 
    depth_image = 1 - depth_image  

    return depth_image


drawing = False
top_left_pt, bottom_right_pt = (-1, -1), (-1, -1)

def draw_rectangle(event, x, y, flags, param):
    global drawing, top_left_pt, bottom_right_pt
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        top_left_pt = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        bottom_right_pt = (x, y)
        cv2.rectangle(param, top_left_pt, bottom_right_pt, (0, 255, 0), 2)
        cv2.imshow('Draw Bounding Box', param)

def main():
    global top_left_pt, bottom_right_pt
    person = input("Enter the person ID: ").zfill(3)
    file_number = int(input("Enter the file number: "))

    depth_folder = "test_data/test5_output"
    depth_file_path = os.path.join(depth_folder, person, f"{person}_{str(file_number).zfill(2)}_cloud.txt")
    rgb_file_path = os.path.join(depth_folder, person, f"{person}_{str(file_number).zfill(2)}_image.png")

    depth_width, depth_height = 960, 540
    rgb_width, rgb_height = 1920, 1080

    if not os.path.exists(depth_file_path) or not os.path.exists(rgb_file_path):
        print("Files not found.")
        return

    depth_image = read_txt_to_depth_image(depth_file_path, depth_width, depth_height)
    rgb_image = cv2.imread(rgb_file_path)

    # x, y, w, h = map(int, input("Enter the coordinates for the bounding box (x y w h): ").split())

    while True:
        clone = rgb_image.copy()
        cv2.namedWindow('Draw Bounding Box', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Draw Bounding Box', 800, 600)
        cv2.moveWindow('Draw Bounding Box', 0, 0)
        cv2.setMouseCallback('Draw Bounding Box', draw_rectangle, clone)

        while True:
            cv2.imshow('Draw Bounding Box', clone)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        x1, y1 = top_left_pt
        x2, y2 = bottom_right_pt
        w = x2 - x1
        h = y2 - y1

        face_region_rgb = rgb_image[y1:y1+h, x1:x1+w]
        scale_x, scale_y = depth_width / rgb_width, depth_height / rgb_height
        x, y, w, h = int(x1 * scale_x), int(y1 * scale_y), int(w * scale_x), int(h * scale_y)
        face_region_depth = depth_image[y:y+h, x:x+w]
        face_region_depth = apply_transformations(face_region_depth)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(face_region_rgb, cv2.COLOR_BGR2RGB))
        plt.title("Cropped RGB Face Image")

        plt.subplot(1, 2, 2)
        plt.imshow(face_region_depth, cmap='gray')
        plt.title("Cropped Depth Face Image")
        figManager = plt.get_current_fig_manager()
        figManager.window.wm_geometry("800x600+900+0")

        plt.show()

        correct_box = input("Is the bounding box correct? (y/n): ")
        if correct_box.lower() == 'y':
            cv2.destroyAllWindows()
            break

    
    output_rgb_dir = f'test_data/test5_face_result/{person}'
    if not os.path.exists(output_rgb_dir):
        os.makedirs(output_rgb_dir)

    cv2.imwrite(f"{output_rgb_dir}/{person}_{str(file_number).zfill(2)}_rgb.png", face_region_rgb)

    output_dir = f'test_data/test5_face_result/{person}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    depth_values_txt_path = f"{output_dir}/{person}_{str(file_number).zfill(2)}_data.txt"
    np.savetxt(depth_values_txt_path, face_region_depth, fmt='%f')

    transformed_depth_image_path = f"{output_dir}/{person}_{str(file_number).zfill(2)}_depth.png"
    cv2.imwrite(transformed_depth_image_path, (face_region_depth * 255).astype(np.uint8))

if __name__ == "__main__":
    main()
