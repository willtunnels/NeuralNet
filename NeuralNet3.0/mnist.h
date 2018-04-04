#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include "Eigen/Eigen/Dense"
#include <exception>

//the code in the mnist namespace is primarily that of users dariush and mrgloom of stackoverflow (with a few of my own modifications)
//see https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c

namespace mnist {
	using label_t = Eigen::Matrix<uint8_t, Eigen::Dynamic, 1>;

	int reverseInt(int i);
	
	std::array<Eigen::MatrixXf, 3> read_mnist_images(const std::string& full_path, int trainingSize, int testSize, int validationSize);

	std::array<mnist::label_t, 3> read_mnist_labels(const std::string& full_path, int trainingSize, int testSize, int validationSize);
}
