#pragma once
#include "mnist.hpp"
#include <Eigen/Core>
#include <array>
#include <initializer_list>
#include <random>
#include <utility>
#include <vector>

class Network
{
private:
  using mvp_t =
    std::pair<std::vector<Eigen::MatrixXf>, std::vector<Eigen::MatrixXf>>;

  size_t numLayers;
  std::vector<Eigen::Index> sizes;

  std::vector<Eigen::VectorXf> bs;
  std::vector<Eigen::MatrixXf> ws;

  // training, test, validation
  std::array<Eigen::MatrixXf, 3> images;
  std::array<mnist::label_t, 3> labels;

  static std::mt19937& getGen();
  static float genRand();

  static float sigmoid(float val);
  static float sigmoidPrime(float val);
  Eigen::VectorXf deltaL(const Eigen::Ref<const Eigen::VectorXf>& a, uint8_t y);

  void randomizeWeights();
  void permuteData();

  void updateMinibatch(Eigen::Index startBatch,
                       Eigen::Index len,
                       float eta,
                       float lmbda);
  mvp_t fwdAndBackProp(const Eigen::Ref<const Eigen::VectorXf>& x, uint8_t y);

  float totalCost(float lmbda);
  float accuracy();
  Eigen::VectorXf feedFwd(const Eigen::Ref<const Eigen::VectorXf>& act);
  float costFnSigmoid(const Eigen::Ref<const Eigen::VectorXf>& a, uint8_t y);

  void initDeltas(std::vector<Eigen::MatrixXf>& deltaWs,
                  std::vector<Eigen::MatrixXf>& deltaBs);

public:
  Network(std::initializer_list<Eigen::Index> sizes,
          const std::string& imgPath,
          const std::string& labelPath,
          int trainingSize,
          int testSize,
          int validationSize);

  Network(std::initializer_list<Eigen::Index> sizes,
          std::vector<Eigen::VectorXf> bs,
          std::vector<Eigen::MatrixXf> ws,
          std::array<Eigen::MatrixXf, 3> images,
          std::array<mnist::label_t, 3> labels);

  void SGD(int epochs, int batchSize, float eta, float lmbda);
};
