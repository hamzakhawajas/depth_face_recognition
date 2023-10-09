# Depth Face Recognition

## Description

This repository contains code for a face recognition project that uses depth data captured from an RGBD sensor. It calculates dissimilarity between two faces based on the depth information.

## Requirements

- cmake
- Python 3.x

You can also install some Python dependencies using pip:

\`\`\`bash
pip install face_recognition
pip install mediapipe
pip install scipy
\`\`\`

## Installation and Running

1. **Build C++ Project**: Navigate to your project directory and execute the following to build `project34topixel.cpp`:

    \`\`\`bash
    cmake CMakeLists.txt && make && ./bin/project3dtopixel
    \`\`\`

2. **Data Conversion**: Go to the `conversion.ipynb` notebook to convert the dataset for preprocessing and training.

3. **Train and Test Model**: Open `train_test.ipynb` to train and test the model.

## Contribution

Feel free to contribute to this project by submitting a pull request.

## License

This project is open source and available under the [MIT License](LICENSE).
