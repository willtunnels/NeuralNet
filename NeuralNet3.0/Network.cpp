#include "Network.h"
#include <ctime>
#include <cmath>
#include <cassert>
#include <cstdio>
#include <limits>

using namespace Eigen;

std::mt19937& Network::getGen() {
	static std::mt19937 gen(static_cast<unsigned>(std::time(NULL)));
	return gen;
}

float Network::genRand() {
	static std::normal_distribution<float> norm;
	return norm(getGen());
}

float Network::sigmoid(float val) {
	return 1.0f / (1.0f + std::expf(-val));
}

float Network::sigmoidPrime(float val) {
	return val * (1 - val);
}

Eigen::VectorXf Network::deltaL(const Eigen::Ref<const Eigen::VectorXf>& a, uint8_t y) {
	VectorXf ret = a;
	ret[y] -= 1;
	return ret;
}

void Network::randomizeWeights() {
	bs.reserve(numLayers - 1);
	ws.reserve(numLayers - 1);
	for (size_t i = 1; i < numLayers; ++i) {
		Eigen::Index y = sizes[i], x = sizes[i - 1];

		bs.emplace_back(y);
		bs.back() = bs.back().unaryExpr([](float) { return genRand(); });

		ws.emplace_back(y, x);
		ws.back() = ws.back().unaryExpr([x](float) { return genRand() / std::sqrtf(static_cast<float>(x)); });
	}
}

void Network::permuteData() {
	MatrixXf& a = images[0];
	mnist::label_t& y = labels[0];

	Transpositions<Dynamic> perm(y.size()); //y.size() is the same as the number of cols in a
	perm.setIdentity();
	std::shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size(), getGen());

	a = a * perm;
	y = perm * y;
}

void Network::updateMinibatch(Eigen::Index startBatch, Eigen::Index len, float eta, float lmbda) {
	std::vector<Eigen::MatrixXf> deltaWs;
	std::vector<Eigen::MatrixXf> deltaBs;
	initDeltas(deltaWs, deltaBs);

	for (Index i = 0; i < len; ++i) {
		Ref<VectorXf> x = images[0].col(startBatch + i);
		uint8_t y = labels[0][startBatch + i];
		auto[deltaNabBs, deltaNabWs] = fwdAndBackProp(x, y);

		assert(deltaWs.size() == deltaBs.size());
		for (std::size_t j = 0; j < deltaNabBs.size(); ++j) {
			deltaBs[j] += deltaNabBs[j];
			deltaWs[j] += deltaNabWs[j];
		}
	}

	assert(ws.size() == deltaWs.size());
	for (std::size_t j = 0; j < ws.size(); ++j) {
		ws[j] = (1 - eta * (lmbda / len)) * ws[j] - (eta / len) * deltaWs[j];
		bs[j] = bs[j] - (eta / len) * deltaBs[j];
	}
}

Network::mvp_t Network::fwdAndBackProp(const Eigen::Ref<const Eigen::VectorXf>& x, uint8_t y) {
	std::vector<Eigen::MatrixXf> deltaWs;
	std::vector<Eigen::MatrixXf> deltaBs;
	initDeltas(deltaWs, deltaBs);

	//forward
	std::vector<VectorXf> activations;
	activations.emplace_back(x);
	VectorXf* curA = &activations[0];

	for (std::size_t i = 0; i < 2; ++i) {
		activations.emplace_back(ws[i] * *curA + bs[i]);
		curA = &activations.back();
		*curA = curA->unaryExpr(&sigmoid);
	}

	//back
	//output layer
	std::size_t curLayer = 2;
	VectorXf delta = deltaL(activations[curLayer], y);
	deltaBs[1] = delta;
	deltaWs[1] = delta * activations[curLayer - 1].transpose();

	//hidden layer
	curLayer = 1;
	VectorXf sp = activations[curLayer].unaryExpr(&sigmoidPrime);
	delta = (ws[curLayer].transpose() * delta).array() * sp.array();
	deltaBs[0] = delta;
	deltaWs[0] = delta * activations[curLayer - 1].transpose();
	return { deltaBs, deltaWs };
}

float Network::totalCost(float lmbda) {
	float cost = 0.0f;
	std::size_t len = images[2].cols();

	for (std::size_t i = 0; i < len; ++i) {
		Ref<VectorXf> x = images[2].col(i);
		uint8_t y = labels[2][i];
		
		VectorXf a = feedFwd(x);
		cost += costFnSigmoid(a, y) / len;
	}

	float temp = 0.0f;
	for (auto&& w : ws)
		temp += w.squaredNorm();
	cost += 0.5f * (lmbda / len) * temp;

	return cost;
}

float Network::accuracy() {
	int sum = 0;
	std::size_t len = images[2].cols();

	for (std::size_t i = 0; i < len; ++i) {
		Ref<VectorXf> x = images[2].col(i);
		uint8_t y = labels[2][i];

		VectorXf a = feedFwd(x);

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

Eigen::VectorXf Network::feedFwd(const Eigen::Ref<const Eigen::VectorXf>& act) {
	VectorXf ret = act;
	for (std::size_t i = 0; i < 2; ++i) {
		VectorXf z = ws[i] * ret + bs[i];
		ret = z.unaryExpr(&sigmoid);
	}
	return ret;
}

float Network::costFnSigmoid(const Eigen::Ref<const Eigen::VectorXf>& a, uint8_t y) {
	float cost = 0.0f;
	for (Index i = 0; i < a.size(); ++i)
		if (y == i)
			if (a[i] > 0)
				cost -= std::logf(a[i]);
			else
				throw std::runtime_error{ "a[i] <= 0" };
				//cost += 99; //arbitrarily large number
		else
			if (1 - a[i] > 0)
				cost -= std::logf(1 - a[i]);
			else
				throw std::runtime_error{ "1 - a[i] <= 0" };
				//cost += 99; //arbitrarily large number
	return cost;
}

void Network::initDeltas(std::vector<Eigen::MatrixXf>& deltaWs, std::vector<Eigen::MatrixXf>& deltaBs) {
	deltaWs.reserve(ws.size());
	deltaBs.reserve(bs.size());
	for (std::size_t i = 0; i < ws.size(); ++i) {
		deltaWs.emplace_back(MatrixXf::Zero(ws[i].rows(), ws[i].cols()));
		deltaBs.emplace_back(MatrixXf::Zero(bs[i].rows(), bs[i].cols()));
	}
}

Network::Network(std::initializer_list<Eigen::Index> sizes, const std::string& imgPath, const std::string& labelPath, int trainingSize, int testSize, int validationSize) : numLayers(sizes.size()), sizes(sizes)
{
	randomizeWeights();
	images = mnist::read_mnist_images(imgPath, trainingSize, testSize, validationSize);
	labels = mnist::read_mnist_labels(labelPath, trainingSize, testSize, validationSize);
}

Network::Network(std::initializer_list<Eigen::Index> sizes,
	std::vector<Eigen::VectorXf> bs, std::vector<Eigen::MatrixXf> ws,
	std::array<Eigen::MatrixXf, 3> images, std::array<mnist::label_t, 3> labels)
	: numLayers(sizes.size()), sizes(sizes),
	bs(std::move(bs)), ws(std::move(ws)),
	images(std::move(images)), labels(std::move(labels))
{
}

void Network::SGD(int epochs, int batchSize, float eta, float etaFac, float lmbda) {
	Index nTrain = labels[0].size();
	Index nVal = labels[2].size();
	for (int i = 0; i < epochs; ++i) {
		permuteData();
		for (Index startBatch = 0; startBatch + batchSize - 1 < images[0].cols(); startBatch += batchSize) {
			updateMinibatch(startBatch, batchSize, eta, lmbda);
		}
		std::printf("Epoch %d training complete\n", i);
		float cost = totalCost(lmbda);
		std::printf("Cost on validation data: %f\n", cost);
		float acc = accuracy();
		std::printf("Accuracy on validation data: %f\n\n", acc);
//		eta = eta * etaFac;
//		etaFac = (0.5f + etaFac) / 1.5f;
	}
}
