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

	std::vector<std::string> folders = {
		"100", "103", "106", "111", "118", "122", "201", "205", "210", "214", "219", "223", "233",
		"101", "104", "108", "112", "119", "124", "202", "208", "212", "215", "221", "228", "234",
		"102", "105", "109", "113", "121", "200", "203", "209", "213", "217", "222", "231" };

	for (auto && folder : folders)
	{
		std::cout << "********************** NEW FOLDER: " + folder + " ***********************" << std::endl;

		FileLoader::load("..\\..\\new_data\\" + folder + "\\data.txt",
			"..\\..\\new_data\\" + folder + "\\label.txt",
			teach_data_inputs, test_data_inputs,
			teach_data_outputs, test_data_outputs,
			1, 0.7);

		std::vector<test_params> tests = {
			{ 10, 15000, 0.001, "quadratic" },
			{ 10, 15000, 0.001, "linear" }
		};

		for (auto && test : tests)
		{
			t_start = static_cast<double>(timer.getTimeMicroseconds()) / 1000.0;

			double C = test.C;
			int max_it = test.MaxIt; // max iterations
			double eps = test.Eps; // stop algorithm when error is below this value

			std::cout << std::endl << "------------------------NEW TEST----------------------" << std::endl;
			std::cout << "C: " << C << " Max it: " << max_it << " Eps: " << eps << std::endl;

			int N = teach_data_inputs.rows();
			int M = teach_data_inputs.cols();

			Eigen::VectorXi classify_outputs;

			MultiSVM svm_instance(test.SVMType, false);

			t_train = static_cast<double>(timer.getTimeMicroseconds()) / 1000.0;
			if ( svm_instance.Train(teach_data_inputs, teach_data_outputs, C, max_it, eps) == false)
				continue;

			t_classify = static_cast<double>(timer.getTimeMicroseconds()) / 1000.0;
			classify_outputs = svm_instance.Classify(test_data_inputs);
			std::cout << "Classyfing complete" << std::endl;

			t_end = static_cast<double>(timer.getTimeMicroseconds()) / 1000.0;

			if (N < 15)
			{
				std::cout << "teach_data_outputs: " << test_data_outputs << std::endl;
				std::cout << "classify_outputs: " << classify_outputs << std::endl;
			}
			else
			{
				std::string path = "..\\..\\tests_results\\reused\\reused_" + folder;
				path += test.SVMType == "quadratic" ? "quad" : "lin";
				path += "_c_" + std::to_string((int)test.C);
				path += "_iter_" + std::to_string((int)test.MaxIt);
				system(("mkdir " + path).c_str());

				std::ofstream correct_file{ path + "\\correct_outputs.txt" };
				std::ofstream classified_file{ path + "\\classified_outputs.txt" };
				std::ofstream times_file(path + "\\times.txt");

				times_file << t_end - t_start << std::endl << t_classify - t_train << std::endl << t_end - t_classify;

				int non_1 = 0;
				int errors = 0;
				int non_1_errors = 0;
				for (int i = 0; i < test_data_inputs.rows(); i++)
				{
					correct_file << test_data_outputs(i) << std::endl;
					classified_file << classify_outputs(i) << std::endl;

					if (test_data_outputs(i) != 1)
					{
						non_1++;
						//std::cout << "should be " << teach_data_outputs(i) << ", is " << classify_outputs(i) << std::endl;
					}

					if (test_data_outputs(i) - classify_outputs(i) != 0)
					{
						errors++;
						if (test_data_outputs(i) != 1)
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
	}
	return 0;
}

