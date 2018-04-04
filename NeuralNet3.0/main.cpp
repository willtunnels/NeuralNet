#include "Network.h"

int main() {
	Network n{ {784, 150, 10}, R"(C:\Users\DonAd\source\repos\NeuralNet3.0\data\training_images)", R"(C:\Users\DonAd\source\repos\NeuralNet3.0\data\training_labels)", 50000, 0, 10000 };
	n.SGD(10, 30, 0.55f, 0.01f);
	n.SGD(10, 30, 0.25f, 0.01f);

	std::printf("done...");
	std::getchar();
}
