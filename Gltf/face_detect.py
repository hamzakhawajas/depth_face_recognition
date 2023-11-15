import os
import json
import base64
import numpy as np
from plyfile import PlyData
import io
import cv2
import matplotlib.pyplot as plt  # NEW: for visualization
from plyfile import PlyElement, PlyData
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
top_left_pt, bottom_right_pt = (-1, -1), (-1, -1)

def detect_faces_dnn(image):
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    return detections

def crop_image_to_points(input_path, depth_dim, threshold=0):
    image = cv2.imread(input_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binarized = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)    
    num_labels, labels = cv2.connectedComponents(binarized)
    x_min = image.shape[1]
    y_min = image.shape[0]
    x_max = 0
    y_max = 0
    for label in range(1, num_labels):  
        points = np.argwhere(labels == label)
        y_coords, x_coords = zip(*points)        
        x_min = min(x_min, min(x_coords))
        x_max = max(x_max, max(x_coords))
        y_min = min(y_min, min(y_coords))
        y_max = max(y_max, max(y_coords))

    if x_min >= x_max or y_min >= y_max:
        return image

    cropped_image = image[y_min:y_max+1, x_min:x_max+1]

    if cropped_image.shape[:2] != depth_dim:
        cropped_image = cv2.resize(cropped_image, (depth_dim[1], depth_dim[0]))

    return cropped_image


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

    
    print(plydata)
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


    width = max_x - min_x + 1
    height = max_y - min_y + 1

    print("image_size: ", img_size_y, img_size_x)
    print("height, width: ", img_size_y,"," ,img_size_x)

    if 'vertex' in plydata:
        vertex_element = plydata['vertex']
        x = vertex_element['x']
        y = vertex_element['y']
        z = vertex_element['z']

        print("x: ", x)
        print("y: ", y)
        print("z: ", z)


        x_mino = x.min()
        x_maxo = x.max()
        y_mino = y.min()
        y_maxo = y.max()

        x_range = x_maxo - x_mino
        y_range = y_maxo - y_mino

        given_width = height
        given_height = width
    
        x_scale = given_width / x_range
        y_scale = given_height / y_range

        x_scaled = ((x - x_mino) * x_scale).astype(int)
        y_scaled = ((y - y_mino) * y_scale).astype(int)



        print("x_scaled: ", x_scaled)
        print("y_scaled: ", y_scaled)

        x_min, x_max = x_scaled.min(), x_scaled.max()
        y_min, y_max = y_scaled.min(), y_scaled.max()

        print("xmin : ", x_min, " xmax: ", x_max )
        print("ymin : ", y_min, " ymax: ", y_max )

        # Now create the depth array with proper dimensions
        depth = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.float32)

        for i in range(len(x)):
            x_idx = x_scaled[i] - x_min
            y_idx = y_scaled[i] - y_min
            depth[y_idx, x_idx] = z[i]

        original_depth = np.copy(depth)
        original_depth = np.flipud(original_depth)
        depth_normalized = ((depth - np.nanmin(depth)) / (np.nanmax(depth) - np.nanmin(depth))) * 255
        depth_normalized = depth_normalized.astype(np.uint8)
        depth_normalized = np.flipud(depth_normalized)
        

    rgb_file_path = file_path.replace('.gltf', '.png')
    rgb = crop_image_to_points(rgb_file_path, original_depth.shape)
    if os.path.exists(rgb_file_path):
        rgb_image = rgb
        rgb_image = cv2.flip(rgb_image, 1)
        detections = detect_faces_dnn(rgb_image)
        
        faces = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  
                box = detections[0, 0, i, 3:7] * np.array([rgb_image.shape[1], rgb_image.shape[0], rgb_image.shape[1], rgb_image.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")
                faces.append((startX, startY, endX, endY))
                cv2.rectangle(rgb_image, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
        # plt.title("Face Detection on RGB Image")
        # plt.show()
        # Check if any face is detected
        if not faces:
            print("No face detected. Skipping this image.")
            return
        # depth_width, depth_height = original_depth.shape  # new provided dimensions for the depth image
        # rgb_width, rgb_height = rgb_image.shape[0], rgb_image.shape[1]      # new provided dimensions for the RGB image
        # scale_x = depth_width / rgb_width
        # scale_y = depth_height / rgb_height

        x, y, w, h = faces[0]
        w = w - x  
        h = h - y 
        # x, y, w, h = int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)

        # x = min(max(x, 0), depth_width)
        # y = min(max(y, 0), depth_height)
        # w = min(w, depth_width - x)
        # h = min(h, depth_height - y)

        face_region_depth = original_depth[y:y+h, x:x+w]


        # Write the point cloud data to a PLY file



        # plt.imshow(original_depth, cmap='gray')
        # plt.title("Original Depth Image")
        # plt.show()
        # Check if the cropped region is empty or not
        if face_region_depth.size == 0:
            print("Cropped face region is empty. Skipping this image.")
            return

        # Display the cropped depth face region
        # plt.imshow(face_region_depth, cmap='gray')
        # plt.title("Cropped Depth Face Image")
        # plt.show()

    else:
        print("Corresponding RGB image not found. Skipping this depth map.")
        return
    

    print(face_region_depth)
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

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        os.makedirs(os.path.join(save_folder, "depth_images"))
        os.makedirs(os.path.join(save_folder, "depth_numpy"))

    for person_folder in sorted(os.listdir(root_folder)):
        person_path = os.path.join(root_folder, person_folder)
        if os.path.isdir(person_path):
            person_count = person_folder

            for gltf_file in sorted(os.listdir(person_path)):
                if gltf_file.endswith('.gltf'):
                    gltf_count = gltf_file.split('.')[0]
                    file_path = os.path.join(person_path, gltf_file)
                    process_gltf_file(file_path, person_count, gltf_count, save_folder)