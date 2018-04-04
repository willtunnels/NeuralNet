#include "Network.h"

Eigen::MatrixXf readInitFile(const std::string& path, Eigen::Index rows, Eigen::Index cols) {
	using namespace Eigen;
	std::ifstream file(path);
	if (file.is_open()) {
		MatrixXf ret(rows, cols);
		for (Index i = 0; i < rows; ++i)
			for (Index j = 0; j < cols; ++j)
				file >> ret(i, j);
		return ret;
	}
	else
		throw std::runtime_error("Unable to open " + path);
}

int main() {
	//Network n{ {784, 30, 10}, R"(C:\Users\DonAd\source\repos\NeuralNet3.0\data\training_images)", R"(C:\Users\DonAd\source\repos\NeuralNet3.0\data\training_labels)", 50000, 0, 10000 };
	//n.SGD( 1, 20, 0.55f, 1.0f, 0.01f);

	std::vector<Eigen::MatrixXf> ws = { readInitFile(R"(G:\Neural\weights0Dim30x784.ccc)", 30, 784),
										readInitFile(R"(G:\Neural\weights1Dim10x30.ccc)", 10, 30) };

	std::vector<Eigen::VectorXf> bs = { readInitFile(R"(G:\Neural\biases0Dim30.ccc)", 30, 1),
										readInitFile(R"(G:\Neural\biases1Dim10.ccc)", 10, 1) };

	auto images = mnist::read_mnist_images(R"(C:\Users\DonAd\source\repos\NeuralNet3.0\data\training_images)", 50000, 0, 10000);
	auto labels = mnist::read_mnist_labels(R"(C:\Users\DonAd\source\repos\NeuralNet3.0\data\training_labels)", 50000, 0, 10000);

	Network n{ {784, 20, 10}, std::move(bs), std::move(ws), std::move(images), std::move(labels) };
	n.SGD(5, 20, 0.55f, 0.0f, 0.01f);

	std::printf("done...");
	std::getchar();
}
