#ifndef ALGORITHM_FILELOADER_H_
#define ALGORITHM_FILELOADER_H_

//FileLoader by Wojciech Gumu³a

//modified data types for our usage 

#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <vector>
#include <iterator>
#include <cmath>

#include "Eigen/Dense"
#include "MultiSVM.h"


class FileLoader {
public:
	static void load(const std::string &data_file, const std::string &label_file,
			  Matrix_T &train_data, Matrix_T &test_data,
			  Class_Vector_T &train_label, Class_Vector_T &test_label,
			  const int skip_first_n_cols, const double split_ratio) {
		Matrix_T data_tmp;
		Class_Vector_T label_tmp;
		load_data(data_file, data_tmp, skip_first_n_cols);
		load_label(label_file, label_tmp);
		normalize(data_tmp);
		split(data_tmp, label_tmp, split_ratio,
			  train_data, test_data, train_label, test_label);
	}
private:
	static void load_data(const std::string &filename, Matrix_T &output, const int skip_first_n_cols = 0) {
		std::ifstream input_stream{filename};
		assert(input_stream.is_open());

		std::string line;
		std::vector<std::vector<double>> file_content;

		int known_width = -1;
		while (std::getline(input_stream, line)) {
			std::stringstream stream{line};
			std::istream_iterator<double> start(stream), end;
			std::advance(start, skip_first_n_cols);

			std::vector<double> line_content(start, end);

			if (known_width < 0)
				known_width = static_cast<int>(line_content.size());
			else
				assert(known_width == static_cast<int>(line_content.size()));

			file_content.push_back(line_content);
		}

		output.resize(file_content.size(), known_width);
		int row = 0;
		int col = 0;
		for (auto &line_content : file_content) {
			for (auto &element : line_content)
				output(row, col++) = element;
			row++;
			col = 0;
		}
	}

	static void load_label(const std::string &filename, Class_Vector_T &output) {
		std::ifstream input_stream{filename};
		assert(input_stream.is_open());

		std::istream_iterator<int> start(input_stream), end;
		std::vector<int> file_content(start, end);

		output.resize(file_content.size(), 1);
		int i = 0;
		for (auto &element : file_content)
			output(i++) = element;
	}

	static void split(const Matrix_T &data, const Class_Vector_T &label,
				      double split_ratio,
					  Matrix_T &data_out1, Matrix_T &data_out2,
					  Class_Vector_T &label_out1, Class_Vector_T &label_out2) {
		assert(split_ratio >= 0 && split_ratio <= 1);
		assert(data.rows() == label.rows());
		int samples = static_cast<int>(data.rows());
		int sample_size = static_cast<int>(data.cols());

		int samples1 = static_cast<int>(samples * split_ratio);
		int samples2 = samples - samples1;

		data_out1.resize(samples1, sample_size);
		data_out1 << data.block(0, 0, samples1, sample_size);

		data_out2.resize(samples2, sample_size);
		data_out2 << data.block(samples1, 0, samples2, sample_size);

		label_out1.resize(samples1, 1);
		label_out1 << label.block(0, 0, samples1, 1);

		label_out2.resize(samples2, 1);
		label_out2 << label.block(samples1, 0, samples2, 1);
	}

	static void normalize(Matrix_T &data) {
		const unsigned int stdev_degrees_of_freedom = 1; // matlab-compatible result
		for (auto col = 0; col < data.cols(); col++) {
			auto data_column = data.col(col);
			const auto mean = data_column.mean();
			data_column.array() -= mean;
			const auto sample_num = data_column.rows();
			const auto stdev = std::sqrt((data_column.array()).square().sum() / (sample_num - stdev_degrees_of_freedom));
			data_column.array() /= stdev;
		}
	}
};

#endif /* ALGORITHM_FILELOADER_H_ */
