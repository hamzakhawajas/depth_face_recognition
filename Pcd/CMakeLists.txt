# cmake_minimum_required(VERSION 2.8)
# project(example)
# find_package(PCL REQUIRED)
# find_package(OpenCV REQUIRED)

# add_definitions(${PCL_DEFINITIONS})
# include_directories(${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
# link_directories(${PCL_LIBRARY_DIRS} ${OpenCV_LIB_DIR})

# SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# add_executable (project3dtopixel src/project3dtopixel.cpp)
# target_link_libraries(project3dtopixel ${PCL_LIBRARIES} ${OpenCV_LIBS})


cmake_minimum_required(VERSION 2.8)
project(example)

# Find the required packages (PCL and OpenCV)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)

# Find Eigen
find_package(Eigen3 REQUIRED)

# Add definitions and include directories
add_definitions(${PCL_DEFINITIONS})
include_directories(
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}  # Include Eigen3
)

# Link directories
link_directories(${PCL_LIBRARY_DIRS} ${OpenCV_LIB_DIR})

# Set the output directory for the executable
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# Create the executable
add_executable (project3dtopixel src/project3dtopixel.cpp)

# Link libraries (PCL, OpenCV, and Eigen)
target_link_libraries(project3dtopixel ${PCL_LIBRARIES} ${OpenCV_LIBS} Eigen3::Eigen)

