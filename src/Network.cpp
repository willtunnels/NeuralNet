#include "Network.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <limits>

using namespace Eigen;

namespace {

std::mt19937&
randGen()
{
  thread_local std::mt19937 gen{ std::random_device{}() };
  return gen;
}

float
normalRand()
{
  thread_local std::normal_distribution<float> norm{};
  return norm(randGen());
}

float
sigmoid(float val)
{
  return 1.0f / (1.0f + std::exp(-val));
}

float
sigmoidPrime(float val)
{
  return val * (1 - val);
}

Eigen::VectorXf
deltaL(const Eigen::Ref<const Eigen::VectorXf>& a, uint8_t y)
{
  VectorXf ret = a;
  ret[y] -= 1;
  return ret;
}

float
costFnSigmoid(const Eigen::Ref<const Eigen::VectorXf>& a, uint8_t y)
{
  float cost = 0.0f;
  for (Index i = 0; i < a.size(); ++i) {
    if (y == i) {
      if (a[i] > 0) {
        cost -= std::log(a[i]);
      } else {
        cost += 99; // arbitrarily large number
      }
    } else {
      if (1 - a[i] > 0) {
        cost -= std::log(1 - a[i]);
      } else {
        cost += 99; // arbitrarily large number
      }
    }
  }
  return cost;
}

} // anonymous namespace

void
Network::randomizeWeights_()
{
  bs_.reserve(numLayers_ - 1);
  ws_.reserve(numLayers_ - 1);
  for (size_t i = 1; i < numLayers_; ++i) {
    Eigen::Index y = sizes_[i], x = sizes_[i - 1];

    bs_.emplace_back(y);
    bs_.back() = bs_.back().unaryExpr([](float) { return normalRand(); });

    ws_.emplace_back(y, x);
    ws_.back() = ws_.back().unaryExpr(
      [x](float) { return normalRand() / std::sqrt(static_cast<float>(x)); });
  }
}

void
Network::permuteData_()
{
  MatrixXf& a = images_[0];
  mnist::label_t& y = labels_[0];

  // y.size() is the same as the number of cols in a
  Transpositions<Dynamic> perm(y.size());
  perm.setIdentity();
  std::shuffle(perm.indices().data(),
               perm.indices().data() + perm.indices().size(),
               randGen());

  a = a * perm;
  y = perm * y;
}

void
Network::initDeltas_(std::vector<Eigen::MatrixXf>& deltaWs,
                     std::vector<Eigen::MatrixXf>& deltaBs)
{
  deltaWs.reserve(ws_.size());
  deltaBs.reserve(bs_.size());
  for (std::size_t i = 0; i < ws_.size(); ++i) {
    deltaWs.emplace_back(MatrixXf::Zero(ws_[i].rows(), ws_[i].cols()));
    deltaBs.emplace_back(MatrixXf::Zero(bs_[i].rows(), bs_[i].cols()));
  }
}

Network::mvp_t
Network::fwdAndBackProp_(const Eigen::Ref<const Eigen::VectorXf>& x, uint8_t y)
{
  std::vector<Eigen::MatrixXf> deltaWs;
  std::vector<Eigen::MatrixXf> deltaBs;
  initDeltas_(deltaWs, deltaBs);

  // forward
  std::vector<VectorXf> activations;
  activations.emplace_back(x);
  VectorXf* curA = &activations[0];

  for (std::size_t i = 0; i < 2; ++i) {
    activations.emplace_back(ws_[i] * *curA + bs_[i]);
    curA = &activations.back();
    *curA = curA->unaryExpr(&sigmoid);
  }

  // back
  // output layer
  std::size_t curLayer = 2;
  VectorXf delta = deltaL(activations[curLayer], y);
  deltaBs[1] = delta;
  deltaWs[1] = delta * activations[curLayer - 1].transpose();

  // hidden layer
  curLayer = 1;
  VectorXf sp = activations[curLayer].unaryExpr(&sigmoidPrime);
  delta = (ws_[curLayer].transpose() * delta).array() * sp.array();
  deltaBs[0] = delta;
  deltaWs[0] = delta * activations[curLayer - 1].transpose();
  return { deltaBs, deltaWs };
}

void
Network::updateMinibatch_(Eigen::Index startBatch,
                          Eigen::Index len,
                          float eta,
                          float lambda)
{
  std::vector<Eigen::MatrixXf> deltaWs;
  std::vector<Eigen::MatrixXf> deltaBs;
  initDeltas_(deltaWs, deltaBs);

  for (Index i = 0; i < len; ++i) {
    Ref<VectorXf> x = images_[0].col(startBatch + i);
    uint8_t y = labels_[0][startBatch + i];
    auto [deltaNabBs, deltaNabWs] = fwdAndBackProp_(x, y);

    assert(deltaWs.size() == deltaBs.size());
    for (std::size_t j = 0; j < deltaNabBs.size(); ++j) {
      deltaBs[j] += deltaNabBs[j];
      deltaWs[j] += deltaNabWs[j];
    }
  }

  assert(ws_.size() == deltaWs.size());
  for (std::size_t j = 0; j < ws_.size(); ++j) {
    ws_[j] = (1 - eta * (lambda / len)) * ws_[j] - (eta / len) * deltaWs[j];
    bs_[j] = bs_[j] - (eta / len) * deltaBs[j];
  }
}

Eigen::VectorXf
Network::feedFwd_(const Eigen::Ref<const Eigen::VectorXf>& act)
{
  VectorXf ret = act;
  for (std::size_t i = 0; i < 2; ++i) {
    VectorXf z = ws_[i] * ret + bs_[i];
    ret = z.unaryExpr(&sigmoid);
  }
  return ret;
}

float
Network::totalCost_(float lambda)
{
  float cost = 0.0f;
  std::size_t len = images_[2].cols();

  for (std::size_t i = 0; i < len; ++i) {
    Ref<VectorXf> x = images_[2].col(i);
    uint8_t y = labels_[2][i];

    VectorXf a = feedFwd_(x);
    cost += costFnSigmoid(a, y) / len;
  }

  float temp = 0.0f;
  for (auto&& w : ws_)
    temp += w.squaredNorm();
  cost += 0.5f * (lambda / len) * temp;

  return cost;
}

float
Network::accuracy_()
{
  int sum = 0;
  std::size_t len = images_[2].cols();

  for (std::size_t i = 0; i < len; ++i) {
    Ref<VectorXf> x = images_[2].col(i);
    uint8_t y = labels_[2][i];

    VectorXf a = feedFwd_(x);

    Index idx = 0;
    float max = a[0];
    for (Index j = 1; j < a.size(); ++j) {
      if (a[j] > max) {
        max = a[j];
        idx = j;
      }
    }
    if (y == idx)
      sum += 1;
  }

  return static_cast<float>(sum) / len;
}

Network::Network(std::initializer_list<Eigen::Index> sizes,
                 const std::string& imgPath,
                 const std::string& labelPath,
                 int trainingSize,
                 int testSize,
                 int validationSize)
  : numLayers_(sizes.size())
  , sizes_(sizes)
{
  randomizeWeights_();
  images_ =
    mnist::read_mnist_images(imgPath, trainingSize, testSize, validationSize);
  labels_ =
    mnist::read_mnist_labels(labelPath, trainingSize, testSize, validationSize);
}

Network::Network(std::initializer_list<Eigen::Index> sizes,
                 std::vector<Eigen::VectorXf> bs,
                 std::vector<Eigen::MatrixXf> ws,
                 std::array<Eigen::MatrixXf, 3> images,
                 std::array<mnist::label_t, 3> labels)
  : numLayers_(sizes.size())
  , sizes_(sizes)
  , bs_(std::move(bs))
  , ws_(std::move(ws))
  , images_(std::move(images))
  , labels_(std::move(labels))
{}

void
Network::SGD(int epochs, int batchSize, float eta, float lambda)
{
  Index nTrain = labels_[0].size();
  Index nVal = labels_[2].size();
  for (int i = 0; i < epochs; ++i) {
    permuteData_();
    for (Index startBatch = 0; startBatch + batchSize - 1 < images_[0].cols();
         startBatch += batchSize) {
      updateMinibatch_(startBatch, batchSize, eta, lambda);
    }
    std::printf("Epoch %d training complete\n", i);
    float cost = totalCost_(lambda);
    std::printf("Cost on validation data: %f\n", cost);
    float acc = accuracy_();
    std::printf("Accuracy on validation data: %f\n\n", acc);
  }
}
