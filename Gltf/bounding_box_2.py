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

    
    #print(plydata)
    import numpy as np

    #projection matrix
    P = np.array([[1052.667867276341, 0, 962.4130834944134, 0],
                [0, 1052.020917785721, 536.2206151001486, 0],
                [0, 0, 1, 0]])

    scale_factor = 1000
    downsample_factor = 1000

    if 'vertex' in plydata:


        vertex_element = plydata['vertex']
        x = vertex_element['x']
        y = vertex_element['y']
        z = vertex_element['z']

        points_3d = np.array([x, y, z, np.ones_like(x)])
        points_2d_hom = np.dot(P, points_3d)
        points_2d = points_2d_hom[:2, :] / points_2d_hom[2, :]

        x_proj = points_2d[0, :]
        y_proj = points_2d[1, :]

        # Round to reduce the range
        x_scaled = np.round(x_proj * scale_factor).astype(int)
        y_scaled = np.round(y_proj * scale_factor).astype(int)

        x_scaled = (x_scaled // downsample_factor)
        y_scaled = (y_scaled // downsample_factor)

        x_min, x_max = x_scaled.min(), x_scaled.max()
        y_min, y_max = y_scaled.min(), y_scaled.max()

        depth = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.float32)

        for i in range(len(x)):
            x_idx = x_scaled[i] - x_min
            y_idx = y_scaled[i] - y_min
            depth[y_idx, x_idx] = z[i]

        print("Depth array created with shape:", depth.shape)

        
        original_depth = np.copy(depth)
        original_depth = np.flipud(original_depth)
        depth_normalized = ((depth - np.nanmin(depth)) / (np.nanmax(depth) - np.nanmin(depth))) * 255
        depth_normalized = depth_normalized.astype(np.uint8)
        depth_normalized = np.flipud(depth_normalized)
        depth_colored = plt.cm.magma(depth_normalized)
    
    depth_colored = (depth_colored[:, :, :3] * 255).astype(np.uint8)
    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)
    clone = depth_colored.copy()
    cv2.namedWindow('Draw Bounding Box', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Draw Bounding Box', 800, 600)
    cv2.setMouseCallback('Draw Bounding Box', draw_rectangle, clone)
    # clone = depth_normalized.copy()
    # cv2.namedWindow('Draw Bounding Box', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Draw Bounding Box', 800, 600)
    # cv2.setMouseCallback('Draw Bounding Box', draw_rectangle, clone)

    while True:
        cv2.imshow('Draw Bounding Box', clone)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    x1, y1 = top_left_pt
    x2, y2 = bottom_right_pt
    w = x2 - x1
    h = y2 - y1
    face_region_depth = original_depth[y1:y1+h, x1:x1+w]
    
    # cropped_original = original_depth[y1:y2, x1:x2]
    # cropped_normalized = normalized_depth[y1:y2, x1:x2]

    while True:  
            plt.imshow(face_region_depth, cmap='magma')
            plt.title("Cropped Depth Face Image")
            plt.show()

            user_input = input("Are you satisfied with the bounding box? (y)/n): ")
            if user_input.lower() == 'y':
                break 
            elif user_input.lower() == 'n':
                # clone = depth_normalized.copy()
                # cv2.namedWindow('Draw Bounding Box', cv2.WINDOW_NORMAL)
                # cv2.resizeWindow('Draw Bounding Box', 800, 600)
                # cv2.setMouseCallback('Draw Bounding Box', draw_rectangle, clone)
                depth_colored = plt.cm.magma(depth_normalized)
                depth_colored = (depth_colored[:, :, :3] * 255).astype(np.uint8)
                depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)
                clone = depth_colored.copy()
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

    # #Display the normalized depth image using OpenCV
    # cv2.imshow('Depth Image', depth_normalized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print(face_region_depth)
    depth_image_folder = os.path.join(save_folder, str(person_count).zfill(3))
    depth_numpy_folder = os.path.join(save_folder, str(person_count).zfill(3))
    if not os.path.exists(depth_image_folder):
        os.makedirs(depth_image_folder)

    depth_image_path = os.path.join(depth_image_folder, f"{str(person_count).zfill(3)}_{str(gltf_count).zfill(2)}_depth_image.png")
    plt.imsave(depth_image_path, face_region_depth, cmap='magma')
    print(depth_image_path)
    if not os.path.exists(depth_numpy_folder):
        os.makedirs(depth_numpy_folder)

    depth_numpy_path = os.path.join(depth_numpy_folder, f"{str(person_count).zfill(3)}_{str(gltf_count).zfill(2)}_depth_data.npy")
    np.save(depth_numpy_path, face_region_depth)


if __name__ == "__main__":
    root_folder = "new_entries/test" 
    save_folder = "new_entries/dataset_numpy_test" 

    for person_folder in sorted(os.listdir(root_folder)):
        person_path = os.path.join(root_folder, person_folder)
        if os.path.isdir(person_path):
            person_count = person_folder

            for gltf_file in sorted(os.listdir(person_path)):
                if gltf_file.endswith('.gltf'):
                    gltf_count = gltf_file.split('.')[0]
                    file_path = os.path.join(person_path, gltf_file)
                    process_gltf_file(file_path, person_count, gltf_count, save_folder)