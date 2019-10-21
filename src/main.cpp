#include "Network.hpp"

int
main(int argc, char** argv)
{
  if (argc != 3) {
    std::printf("usage: neuralnetwork IMAGE_PATH, LABEL_PATH\n");
    return 1;
  }

  Network n{ { 784, 150, 10 }, argv[1], argv[2], 50000, 0, 10000 };
  n.SGD(10, 30, 0.55f, 0.01f);
  n.SGD(10, 30, 0.25f, 0.01f);

  std::printf("exiting...\n");
  return 0;
}
