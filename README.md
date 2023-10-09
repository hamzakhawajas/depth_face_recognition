# Depth Face Recognition

## Description

This repository contains code for a face recognition project that uses depth data captured from an RGBD sensor. It calculates dissimilarity between two faces based on the depth information.

## Installation and Running
1. `pip install requirements.txt`

2. **Build C++ Project**: Navigate to your project directory and execute the following to build project3dtopixel.cpp:

   `cmake CMakeLists.txt && make && ./bin/project3dtopixel`

2. **Data Conversion**: Go to the `conversion.ipynb` notebook to convert and preprocess the dataset

3. **Train and Test Model**: Open `train_test.ipynb` to train and test the model.

