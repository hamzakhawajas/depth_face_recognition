#include <iostream>
#include <fstream>
#include <filesystem>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>
#include <unistd.h>
#include <vector>
#include <algorithm>

using namespace std;
namespace fs = std::filesystem;

int main() {
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        cout << "Current working directory: " << cwd << endl;
    } else {
        perror("getcwd() error");
        return 1;
    }

    // Prepare a vector to store filenames
    vector<string> filenames;
    for (const auto& entry : fs::directory_iterator("RGBD_Face_dataset_testing/test5")) {
        filenames.push_back(entry.path().filename().string());
    }

    // Sort the filenames
    sort(filenames.begin(), filenames.end());

    for (const auto& filename : filenames) {
        // Skip if not a PCD file
        if (filename.substr(filename.find_last_of(".") + 1) != "pcd") {
            continue;
        }

        // Extract person and file number from filename
        string filename_base = filename.substr(0, filename.find_last_of("."));
        string person_str = filename_base.substr(0, 3);
        string file_num_str = filename_base.substr(4, 2);

        cout << "Processing " << filename_base << " for person " << person_str << " and file number " << file_num_str << endl;

        // Full path for both files
        string pcd_file = "RGBD_Face_dataset_testing/test5/" + filename_base + ".pcd";
        string png_file = "RGBD_Face_dataset_testing/test5/" + person_str + "_" + file_num_str + "_image.png";

        // Check if the files exist
        if (!fs::exists(pcd_file) || !fs::exists(png_file)) {
            cout << "File does not exist: " << pcd_file << " or " << png_file << endl;
            continue;
        }

        // Create a sub-directory for each person
        fs::path person_dir("test_data/test5_output/" + person_str);
        fs::create_directories(person_dir);

        // Open a text file to write
        ofstream file(person_dir.string() + "/" + filename_base + "_cloud.txt");

        // Load point cloud
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(pcd_file, *cloud) == -1) {
            cout << "Couldn't read .pcd file: " << pcd_file << endl;
            continue;
        }

        // Define projection matrix
        Eigen::Matrix<float, 3, 4> P;
        P << 1052.667867276341, 0, 962.4130834944134, 0, 0, 1052.020917785721, 536.2206151001486, 0, 0, 0, 1, 0;

        // Write header
        file << "Width (i)\tHeight (j)\t3D Point X\t3D Point Y\t3D Point Z\t2D Projected X\t2D Projected Y\n";

        for (int i = 0; i < cloud->width; i++) {
            for (int j = 0; j < cloud->height; j++) {
                pcl::PointXYZRGB point = cloud->at(i, j);

                // Project point to 2D
                Eigen::Vector4f homogeneous_point(point.x, point.y, point.z, 1);
                Eigen::Vector3f output = P * homogeneous_point;
                output[0] /= output[2];
                output[1] /= output[2];

                // Write data to text file
                file << i << "\t" << j << "\t" << point.x << "\t" << point.y << "\t" << point.z << "\t" << output[0] << "\t" << output[1] << "\n";
            }
        }

        // Close the text file
        file.close();

        // Copy the image without reading
        fs::copy(png_file, person_dir.string() + "/" + filename_base + "_image.png");
    }

    cout << "Text files and images created in output directories." << endl;

    return 0;
}


