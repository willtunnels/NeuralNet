#ifndef _7a5989c9_1e80_4fb8_b847_b21952220c2f
#define _7a5989c9_1e80_4fb8_b847_b21952220c2f

#include "mnist.hpp"
#include <Eigen/Core>
#include <array>
#include <initializer_list>
#include <random>
#include <utility>
#include <vector>

class Network
{
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

private:
  using mvp_t =
    std::pair<std::vector<Eigen::MatrixXf>, std::vector<Eigen::MatrixXf>>;

  void randomizeWeights_();

  void permuteData_();

  void initDeltas_(std::vector<Eigen::MatrixXf>& deltaWs,
                   std::vector<Eigen::MatrixXf>& deltaBs);

  mvp_t fwdAndBackProp_(const Eigen::Ref<const Eigen::VectorXf>& x, uint8_t y);

  void updateMinibatch_(Eigen::Index startBatch,
                        Eigen::Index len,
                        float eta,
                        float lmbda);

  Eigen::VectorXf feedFwd_(const Eigen::Ref<const Eigen::VectorXf>& act);

  float totalCost_(float lmbda);

  float accuracy_();

  size_t numLayers_;
  std::vector<Eigen::Index> sizes_;
  std::vector<Eigen::VectorXf> bs_;
  std::vector<Eigen::MatrixXf> ws_;
  // training, test, validation
  std::array<Eigen::MatrixXf, 3> images_;
  std::array<mnist::label_t, 3> labels_;
};

#endif
