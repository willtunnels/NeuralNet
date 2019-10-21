#include "mnist.hpp"
#include <array>

int
mnist::reverseInt(int i)
{
  unsigned char c1, c2, c3, c4;
  c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
  return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

std::array<Eigen::MatrixXf, 3>
mnist::read_mnist_images(const std::string& full_path,
                         int trainingSize,
                         int testSize,
                         int validationSize)
{
  using namespace Eigen;
  std::ifstream file(full_path, std::ios::binary);
  if (file.is_open()) {
    int magic_number = 0, n_rows = 0, n_cols = 0;

    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);

    if (magic_number != 2051)
      throw std::runtime_error("Invalid MNIST image file!");

    int actual_nm_imgs;
    file.read((char*)&actual_nm_imgs, sizeof(actual_nm_imgs)),
      actual_nm_imgs = reverseInt(actual_nm_imgs);

    assert(actual_nm_imgs >=
           trainingSize + testSize +
             validationSize); // temporary code for testing purposes
    actual_nm_imgs = trainingSize + testSize + validationSize;

    file.read((char*)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
    file.read((char*)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

    int image_size = n_rows * n_cols;
    uint8_t* buff = new uint8_t[image_size * actual_nm_imgs];
    file.read((char*)buff, image_size * actual_nm_imgs);

    assert(trainingSize + testSize <= actual_nm_imgs);
    std::array<MatrixXf, 3> _dataset = {
      Map<Matrix<uint8_t, Dynamic, Dynamic>>(buff, image_size, trainingSize)
          .cast<float>() /
        256.0f,
      Map<Matrix<uint8_t, Dynamic, Dynamic>>(
        buff + trainingSize * image_size, image_size, testSize)
          .cast<float>() /
        256.0f,
      Map<Matrix<uint8_t, Dynamic, Dynamic>>(
        buff + (trainingSize + testSize) * image_size,
        image_size,
        actual_nm_imgs - trainingSize - testSize)
          .cast<float>() /
        256.0f
    };

    delete[] buff;
    return _dataset;
  } else {
    throw std::runtime_error("Unable to open file " + full_path);
  }
}

std::array<mnist::label_t, 3>
mnist::read_mnist_labels(const std::string& full_path,
                         int trainingSize,
                         int testSize,
                         int validationSize)
{
  using namespace Eigen;
  std::ifstream file(full_path, std::ios::binary);
  if (file.is_open()) {
    int magic_number = 0;

    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);

    if (magic_number != 2049)
      throw std::runtime_error("Invalid MNIST label file!");

    int actual_nm_labels;
    file.read((char*)&actual_nm_labels, sizeof(actual_nm_labels)),
      actual_nm_labels = reverseInt(actual_nm_labels);

    assert(actual_nm_labels >=
           trainingSize + testSize +
             validationSize); // temporary code for testing purposes
    actual_nm_labels = trainingSize + testSize + validationSize;

    uint8_t* buff = new uint8_t[actual_nm_labels];
    file.read((char*)buff, actual_nm_labels);

    assert(trainingSize + testSize <= actual_nm_labels);
    std::array<mnist::label_t, 3> _dataset = {
      Map<mnist::label_t>(buff, trainingSize),
      Map<mnist::label_t>(buff + trainingSize, testSize),
      Map<mnist::label_t>(buff + trainingSize + testSize,
                          actual_nm_labels - trainingSize - testSize)
    };

    delete[] buff;
    return _dataset;
  } else {
    throw std::runtime_error("Unable to open file " + full_path);
  }
}
