# Depth Face Recognition

## Description

This repository contains code for a face recognition project that uses depth data captured from an RGBD sensor. It calculates dissimilarity between two faces based on the depth information.

## Installation and Running
#For PCD files
1. `pip install requirements.txt`

2. **Build C++ Project**: Navigate to your project directory and execute the following to build project3dtopixel.cpp:

   `cmake CMakeLists.txt && make && ./bin/project3dtopixel`

2. **Data Conversion**: Go to the `conversion.ipynb` notebook to convert and preprocess the dataset

3. **Train and Test Model**: Open `train_test.ipynb` to train and test the model.


#For GLTF files
1. Run `face_detect.py` or `bounding_box.py` to generate numpy arrays for face region.

2. Run `padding.py` to transform 2d arrays to eqaul shape of 300x300.

3. Use `training.py` to train the model.

4. Run `generate_embeddings.py` to obatain embeddings.



