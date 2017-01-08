// esdmit_svm.cpp : Defines the entry point for the console application.
//



#include "stdafx.h"
#include "Eigen/Eigen"
#include "BinarySVM.h"
#include "MultiSVM.h"
#include "FileLoader.hpp"

//not ours - timer
#include "util.hpp"

#include <iostream>
#include <fstream>


#define BINARY_SVM_TESTx
#define MULTI_SVM_TESTx
#define MULTI_SVM_TEST_QUAD


struct test_params{
	double C;
	int MaxIt;
	double Eps;
	std::string SVMType;
};

int _tmain(int argc, _TCHAR* argv[])
{
	util::Timer timer;
	double t_start = static_cast<double>(timer.getTimeMicroseconds()) / 1000.0;
	double t_train;
	double t_classify;
	double t_end;

	Matrix_T teach_data_inputs;
	Class_Vector_T teach_data_outputs;
	Matrix_T test_data_inputs;
	Class_Vector_T test_data_outputs;

	FileLoader::load("..\\..\\ReferencyjneDane2\\ConvertedQRSRawData_all.txt",
		"..\\..\\ReferencyjneDane2\\Class_IDs_all.txt",
		teach_data_inputs, test_data_inputs,
		teach_data_outputs, test_data_outputs,
		1, 0.7);

	std::vector<test_params> tests = {
		{ 10, 100, 0.001, "quadratic" },
		{ 100, 150, 0.001, "linear"}
	};

	for (auto && test : tests)
	{
		t_start = static_cast<double>(timer.getTimeMicroseconds()) / 1000.0;

		double C = test.C;
		int max_it = test.MaxIt; // max iterations
		double eps = test.Eps; // stop algorithm when error is below this value
		
		int N = teach_data_inputs.rows();
		int M = teach_data_inputs.cols();

		
		MultiSVM svm_instance(test.SVMType, false);

		t_train = static_cast<double>(timer.getTimeMicroseconds()) / 1000.0;
		svm_instance.Train(teach_data_inputs, teach_data_outputs, C, max_it, eps);// , w0);


		t_classify = static_cast<double>(timer.getTimeMicroseconds()) / 1000.0;
		classify_outputs = svm_instance.Classify(test_data_inputs);


		t_end = static_cast<double>(timer.getTimeMicroseconds()) / 1000.0;



#endif //MULTI_SVM_TEST

#ifdef MULTI_SVM_TEST_QUAD

		Eigen::VectorXi classify_outputs;
		//Eigen::MatrixXd teach_data_inputs(7, 2);
		//Eigen::VectorXi teach_data_outputs(7);



		//teach_data_inputs << -1.3887, -1.3887,
		//	-0.9258, -0.9258,
		//	-0.4629, -0.4629,
		//	0, 0,
		//	0.4629, 0.4629,
		//	0.9258, 0.9258,
		//	1.3887, 1.3887;

		//teach_data_outputs << 1, 1, 2, 2, 2, 1, 1;

		double C = 10;
		int max_it = 10000; // max iterations
		double eps = 0.001; // stop algorithm when error is below this value
		int N = teach_data_inputs.rows();
		int M = teach_data_inputs.cols();

		////Starting point
		Eigen::VectorXd w0(MultiSVM::QuadraticKernelSize(M) + 1);
		w0.setRandom();
		//w0 << 0.6557, 0.0357, 0.8491, 0.9340;


		MultiSVM svm_instance("quadratic", false);

		t_train = static_cast<double>(timer.getTimeMicroseconds()) / 1000.0;
		svm_instance.Train(teach_data_inputs, teach_data_outputs, C, max_it, eps);// , w0);


		t_classify = static_cast<double>(timer.getTimeMicroseconds()) / 1000.0;
		classify_outputs = svm_instance.Classify(test_data_inputs);
		std::cout << "Classyfing complete" << std::endl;

		t_end = static_cast<double>(timer.getTimeMicroseconds()) / 1000.0;
#endif //MULTI_SVM_TEST_QUAD


		if (N < 15)
		{
			std::cout << "teach_data_outputs: " << test_data_outputs << std::endl;
			std::cout << "classify_outputs: " << classify_outputs << std::endl;
		}
		else
		{
			std::ofstream correct_file{ "correct_outputs.txt" };
			std::ofstream classified_file{ "classified_outputs.txt" };

			int non_1 = 0;
			int errors = 0;
			int non_1_errors = 0;
			for (int i = 0; i < test_data_inputs.rows(); i++)
			{
				correct_file << test_data_outputs(i) << std::endl;
				classified_file << classify_outputs(i) << std::endl;

				if (test_data_outputs(i) != 4)
				{
					non_1++;
					//std::cout << "should be " << teach_data_outputs(i) << ", is " << classify_outputs(i) << std::endl;
				}

				if (test_data_outputs(i) - classify_outputs(i) != 0)
				{
					errors++;
					if (test_data_outputs(i) != 4)
					{
						non_1_errors++;
					}
				}
			}
			correct_file.close();
			classified_file.close();

			std::cout << "Inputs: " << test_data_inputs.rows() << "\nerrors: " << errors << "\nnon-one's: " << non_1 << "\nnon-one errors: " << non_1_errors << std::endl;
		}


		std::cout << "Program runtime: " << (t_end - t_start) << " ms" << std::endl;
		std::cout << "Training runtime: " << (t_classify - t_train) << " ms" << std::endl;
		std::cout << "Classifying runtime: " << (t_end - t_classify) << " ms" << std::endl;
	}

	return 0;
}

