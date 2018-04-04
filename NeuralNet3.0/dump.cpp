//Eigen::MatrixXf readInitFile(const std::string& path, Eigen::Index rows, Eigen::Index cols) {
//	using namespace Eigen;
//	std::ifstream file(path);
//	if (file.is_open()) {
//		MatrixXf ret(rows, cols);
//		for (Index i = 0; i < rows; ++i)
//			for (Index j = 0; j < cols; ++j)
//				file >> ret(i, j);
//		return ret;
//	}
//	else
//		throw std::runtime_error("Unable to open " + path);
//}

//Eigen::MatrixXf Network::readWsOrBsFile(const std::string& pathBiases, const std::string& pathWeights, Eigen::Index rows, Eigen::Index cols) {
//	std::ifstream file(pathBiases);
//	Eigen::MatrixXf res(rows, cols);
//	for (Eigen::Index i = 0; i < rows; ++i) {
//		for (Eigen::Index j = 0; j < cols; ++j) {
//			file >> res(i, j);
//		}
//	}
//	return res;
//}

//std::array<mnist::label_t, 3> mnist::read_csv_labels(const std::string& full_path, int trainingSize, int testSize, int validationSize) {
//	using namespace Eigen;
//	std::ifstream file(full_path);
//
//	std::vector<uint8_t> buff;
//	buff.reserve(trainingSize + testSize + validationSize);
//
//	if (file.is_open()) {
//		std::string value;
//		while (getline(file, value, ','))
//			buff.push_back(std::stoi(value));
//
//		std::array<mnist::label_t, 3> _dataset = { Map<mnist::label_t>(buff.data(), trainingSize),
//												   Map<mnist::label_t>(buff.data() + trainingSize, testSize),
//												   Map<mnist::label_t>(buff.data() + trainingSize + testSize, buff.size() - trainingSize - testSize) };
//	}
//	else {
//		throw std::runtime_error("Unable to open file " + full_path);
//	}
//}

//std::array<mnist::label_t, 3> mnist::read_gz_labels(const std::string& full_path, int trainingSize, int testSize, int validationSize) {
//	uint8_t buff[1024 * 1024 * 16]; //16 MiB
//	gzFile fi = gzopen(full_path.c_str(), "rb");
//	gzrewind(fi);
//	while (!gzeof(fi))
//	{
//		int len = gzread(fi, buff, sizeof(buff));
//		//buf contains len bytes of decompressed data
//	}
//	gzclose(fi);
//}
