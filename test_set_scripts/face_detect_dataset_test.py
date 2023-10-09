import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches

def read_txt_to_depth_image(file_path, width, height):
    depth_image = np.zeros((height, width))
    with open(file_path, 'r') as f:
        next(f)  # Skip header line
        for line in f:
            i, j, x, y, z, _, _ = map(float, line.strip().split('\t'))
            if 0 <= int(i) < width and 0 <= int(j) < height:
                depth_image[int(j), int(i)] = z
    return depth_image


import mediapipe as mp


mp_face_detection = mp.solutions.face_detection

def detect_face_mp(image):
    with mp_face_detection.FaceDetection(min_detection_confidence=0.1) as face_detection:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = face_detection.process(rgb_image)
        faces = []
        
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                faces.append([x, y, w, h])
        return faces



import face_recognition

def detect_face_fc(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)

    faces = []
    for (top, right, bottom, left) in face_locations:
        faces.append([left, top, right - left, bottom - top])
    
    return faces


def detect_face_cv(image):
    net = cv2.dnn.readNetFromCaffe("models/face/deploy.prototxt", "models/face/res10_300x300_ssd_iter_140000.caffemodel")
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    net.setInput(blob)
    detections = net.forward()
    
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # you can adjust this threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append([startX, startY, endX-startX, endY-startY])
            
    return faces

def main():
    depth_width, depth_height = 960, 540
    rgb_width, rgb_height = 1920, 1080

    depth_folder = 'test1_output'
    depth_files = [f for f in os.listdir(depth_folder) if f.endswith('_cloud.txt')]

    for depth_file in depth_files:
        person, file_number_str = depth_file.split('_')[:2]
        file_number = int(file_number_str)
        person = int(person)

        print(f"person: {person}  image: {file_number}")

        depth_file_path = os.path.join(depth_folder, depth_file)
        rgb_file_path = f'RGBD_Face_dataset_testing/Test1/{str(person).zfill(3)}_{str(file_number).zfill(2)}_image.png'


        print("depth path: " + str(depth_file_path) + " rgb_file_path: " + str(rgb_file_path))

        if not os.path.exists(depth_file_path) or not os.path.exists(rgb_file_path):
            continue

        depth_image = read_txt_to_depth_image(depth_file_path, depth_width, depth_height)
        rgb_image = cv2.imread(rgb_file_path)


        faces = detect_face_fc(rgb_image)
        if len(faces) == 0:
            faces = detect_face_mp(rgb_image)
            if len(faces) == 0:
                faces

        for (x, y, w, h) in faces:

            cropped_rgb_face = rgb_image[y:y+h, x:x+w]
            
            output_rgb_dir = f'test1_result/'
            if not os.path.exists(output_rgb_dir):
                os.makedirs(output_rgb_dir)
            cv2.imwrite(f"{output_rgb_dir}/{str(person).zfill(3)}_{str(file_number).zfill(2)}_face_rgb.png", cropped_rgb_face)
            
            scale_x, scale_y = depth_width / rgb_width, depth_height / rgb_height
            x, y, w, h = int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)
            face_region_depth = depth_image[y:y+h, x:x+w]

            output_dir = f'test1_result/'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            output_text_path = f"{output_dir}/{str(person).zfill(3)}_{str(file_number).zfill(2)}_face.txt"

            with open(output_text_path, 'w') as f:
                f.write("z-values for each (x, y) in a 2D grid\n")
                for j in range(face_region_depth.shape[0]):  # Vertical (y-axis)
                    for i in range(face_region_depth.shape[1]):  # Horizontal (x-axis)
                        z_value = face_region_depth[j, i]
                        f.write(f"{z_value}\t")
                    f.write("\n")
            
            output_depth_face = f'test1_result/'
            if not os.path.exists(output_depth_face):
                os.makedirs(output_depth_face)
            plt.imsave(f"{output_depth_face}/{str(person).zfill(3)}_{str(file_number).zfill(2)}_face.png", face_region_depth, cmap='gray')


                


if __name__ == "__main__":
    main()
