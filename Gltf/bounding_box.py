import os
import json
import base64
import numpy as np
from plyfile import PlyData
import io
import cv2
import matplotlib.pyplot as plt

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

def process_gltf_file(file_path, person_count, gltf_count, save_folder):

    try:
        with open(file_path, 'r') as f:
            gltf_data = json.load(f)
    except Exception as e:
        print(f"Error loading GLTF: {e}")
        return

    try:
        data_uri = gltf_data["meshes"][0]["extras"]["dataURI"]
    except (KeyError, IndexError) as e:
        print(f"Couldn't find the necessary data in the GLTF file: {e}")
        return

    base64_data = data_uri.split(",")[-1]
    decoded_data = base64.b64decode(base64_data)
    buffer = io.BytesIO(decoded_data)

    try:
        plydata = PlyData.read(buffer)
    except Exception as e:
        print(f"Error reading PLY data: {e}")
        return

    if 'vertex' in plydata:
        vertex_element = plydata['vertex']
        x = vertex_element['x']
        y = vertex_element['y']
        z = vertex_element['z']

        print("x: ", x)
        print("y: ", y)
        print("z: ", z)
    


        all_data = np.vstack((x, y, z)).T
        print("All Data:")
        print(all_data)

        png_file_path = file_path.replace('.gltf', '.png')
        if os.path.exists(png_file_path):
            png_image = cv2.imread(png_file_path)
        
        non_zero_indices = cv2.findNonZero(cv2.cvtColor(png_image, cv2.COLOR_BGR2GRAY))
        
        min_x = np.amin(non_zero_indices[:,:,0])
        max_x = np.amax(non_zero_indices[:,:,0])
        min_y = np.amin(non_zero_indices[:,:,1])
        max_y = np.amax(non_zero_indices[:,:,1])

        print(max_x, min_x)
        print(max_y, min_y)

        img_size_x = max_x - min_x + 1
        img_size_y = max_y - min_y + 1

        print("image_size: ", img_size_y, img_size_x)

        x_img = np.interp(x, (x.min(), x.max()), (0, img_size_x - 1)).astype(int)
        y_img = np.interp(y, (y.min(), y.max()), (0, img_size_y - 1)).astype(int)

        depth = np.full((img_size_y, img_size_x), np.nan, dtype=np.float32)

        depth_threshold = 0.0


        ###################################################################
        # # Create another array to store the raw depth values
        # raw_depth = np.full((img_size_y, img_size_x), np.nan, dtype=np.float32)


        # for i in range(len(x)):
        #     raw_depth[y_img[i], x_img[i]] = z[i]

#####################################################################3

        for i in range(len(x)):
            if z[i] > depth_threshold:
                depth[y_img[i], x_img[i]] = z[i]


        depth = np.flipud(depth)
        depth = np.nan_to_num(depth, nan=np.nanmax(depth))
        original_depth = np.copy(depth)
        depth_normalized = ((depth - np.nanmin(depth)) / (np.nanmax(depth) - np.nanmin(depth))) * 255
        depth_normalized = depth_normalized.astype(np.uint8)

        plt.show()


    clone = depth_normalized.copy()
    cv2.namedWindow('Draw Bounding Box', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Draw Bounding Box', img_size_x, img_size_y)
    cv2.setMouseCallback('Draw Bounding Box', draw_rectangle, clone)

    while True:
        cv2.imshow('Draw Bounding Box', clone)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    x1, y1 = top_left_pt
    x2, y2 = bottom_right_pt
    w = x2 - x1
    h = y2 - y1
    face_region_depth = original_depth[y1:y1+h, x1:x1+w]
    

    while True:
            plt.imshow(face_region_depth, cmap='gray')
            plt.title("Cropped Depth Face Image")
            plt.show()

            user_input = input("Are you satisfied with the bounding box? (y)/n): ")
            if user_input.lower() == 'y':
                break 
            elif user_input.lower() == 'n':
                clone = depth_normalized.copy()
                cv2.namedWindow('Draw Bounding Box', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Draw Bounding Box', 800, 600)
                cv2.setMouseCallback('Draw Bounding Box', draw_rectangle, clone)

                while True:
                    cv2.imshow('Draw Bounding Box', clone)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

    x1, y1 = top_left_pt
    x2, y2 = bottom_right_pt
    w = x2 - x1
    h = y2 - y1
    face_region_depth = original_depth[y1:y1+h, x1:x1+w]


    print("face region depth shape:",face_region_depth.shape)
    print("face region depth:\n",face_region_depth)
    depth_image_folder = os.path.join(save_folder, str(person_count).zfill(3))
    depth_numpy_folder = os.path.join(save_folder, str(person_count).zfill(3))
    if not os.path.exists(depth_image_folder):
        os.makedirs(depth_image_folder)

    depth_image_path = os.path.join(depth_image_folder, f"{str(person_count).zfill(3)}_{str(gltf_count).zfill(2)}_depth_image.png")
    plt.imsave(depth_image_path, face_region_depth, cmap='gray')
    print(depth_image_path)
    if not os.path.exists(depth_numpy_folder):
        os.makedirs(depth_numpy_folder)

    depth_numpy_path = os.path.join(depth_numpy_folder, f"{str(person_count).zfill(3)}_{str(gltf_count).zfill(2)}_depth_data.npy")
    np.save(depth_numpy_path, face_region_depth)


if __name__ == "__main__":
    root_folder = "train" 
    save_folder = "dataset_numpy_train"

    for person_folder in sorted(os.listdir(root_folder)):
        person_path = os.path.join(root_folder, person_folder)
        if os.path.isdir(person_path):
            person_count = person_folder

            for gltf_file in sorted(os.listdir(person_path)):
                if gltf_file.endswith('.gltf'):
                    gltf_count = gltf_file.split('.')[0]
                    file_path = os.path.join(person_path, gltf_file)
                    process_gltf_file(file_path, person_count, gltf_count, save_folder)