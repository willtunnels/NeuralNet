//mnist::label_t trainingL(4);
//trainingL << 0, 1, 2, 3;
//std::array<mnist::label_t, 3> labels = { trainingL, mnist::label_t{}, trainingL };
//Eigen::MatrixXf trainingI(4, 3);
//trainingI << .53f, .98f, .52f, //image 1
//			 .06f, .37f, .97f, //image 2
//			 .59f, .45f, .12f, //image 3
//			 .48f, .94f, .31f; //image 4
//trainingI.transposeInPlace();
//std::array<Eigen::MatrixXf, 3> images = { trainingI, Eigen::MatrixXf{}, trainingI };
//Eigen::VectorXf bs0(2);
//bs0 << -.02f, 1.5f;
//Eigen::VectorXf bs1(4);
//bs1 << -.7f, .78f, .04f, -1.8f;
//std::vector<Eigen::VectorXf> bs{ std::move(bs0), std::move(bs1) };
//Eigen::MatrixXf ws0(2, 3);
//ws0 << -.6f,  .16f,  .19f,
//		.23f, .51f, -.2f;
//Eigen::MatrixXf ws1(4, 2);
//ws1 <<  -.97f,  .42f,
//		 .67f, -.46f,
//		-.75f, -.59f,
//		 .24f, -.44f;
//std::vector<Eigen::MatrixXf> ws{ std::move(ws0), std::move(ws1) };

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
