// esdmit_svm.cpp : Defines the entry point for the console application.
//



#include "stdafx.h"
#include "Eigen/Eigen"
#include "BinarySVM.h"
#include "MultiSVM.h"

//not ours - timer
#include "util.hpp"

#include <iostream>
#include <fstream>


#define BINARY_SVM_TESTx
#define MULTI_SVM_TESTx
#define MULTI_SVM_TEST_QUAD

int _tmain(int argc, _TCHAR* argv[])
{
	util::Timer timer;
	double t_start = static_cast<double>(timer.getTimeMicroseconds()) / 1000.0;
	double t_train;
	double t_classify;
	double t_end;

	Eigen::MatrixXd teach_data_inputs(1068, 18);
	Eigen::VectorXi teach_data_outputs(1068);
	std::ifstream file_data("..\\..\\ReferencyjneDane2\\100\\ConvertedQRSRawData_2.txt", std::ios::in);
	std::ifstream file_class("..\\..\\ReferencyjneDane2\\100\\Class_IDs_2.txt", std::ios::in);

	double num;
	int i = 0;
	int j = 0;
	while (1)
	{
		file_data >> num;
		if (file_data.eof())
			break;

		teach_data_inputs(i, j) = num;
		
		j++;
		if (j >= 18)
		{
			j = 0;
			//std::cout << i++ << std::endl;
		}
	}
	file_data.close();

	i = 0;
	while (1)
	{
		file_class >> num;
		if (file_class.eof())
			break;

		teach_data_outputs(i) = num;

		i++;
	}
	file_class.close();

#ifdef BINARY_SVM_TEST
	Eigen::Matrix<double, 7, 2> teach_data_inputs;
	Eigen::VectorXi teach_data_outputs(7);
	Eigen::VectorXi classify_outputs;

	teach_data_inputs << -1.0874, -1.0874,
		-0.9182, -0.9182,
		-0.4108, -0.4108,
		-0.2416, -0.2416,
		-0.0725, -0.0725,
		1.4498, 1.2807,
		1.2807, 1.4498;

		teach_data_outputs << -1, -1, 1, 1, 1, 1, 1;

	// Paramteter C(0, inf) small C = fuck contraints, large margin big.C = use constrains, smaller margin but less mistakes in training data
	// C = inf enforces no mistakes in training data, not always possible so
	// output may be some random shit
	double C = 10; 
	int max_it = 10000;// 10000; // max iterations
	double eps = 0.001; // stop algorithm when error is below this value
	int N = teach_data_inputs.rows();
	int M = teach_data_inputs.cols();

	//Starting point
	Eigen::VectorXd w0(M+1);
	//w0.setRandom();
	w0 << 0.9649, 0.1576, 0.9706;

	BinarySVM svm_instance;

	t_train = static_cast<double>(timer.getTimeMicroseconds()) / 1000.0;
	svm_instance.Train(teach_data_inputs, teach_data_outputs, w0, C, max_it, eps);
	

	t_classify = static_cast<double>(timer.getTimeMicroseconds()) / 1000.0;
	classify_outputs = svm_instance.Classify(teach_data_inputs);

	t_end = static_cast<double>(timer.getTimeMicroseconds()) / 1000.0;

#endif // BINARY_SVM_TEST
#ifdef MULTI_SVM_TEST

	Eigen::VectorXi classify_outputs;
	//Eigen::Matrix<double, 9, 2> teach_data_inputs;
	//Eigen::VectorXi teach_data_outputs(9);
	


	//teach_data_inputs << 1, 3,
	//					2, 2,
	//					7, 7,
	//					7, 9,
	//					6, 1,
	//					8, 6,
	//					-7, -7,
	//					-7, -9,
	//					1, 1;

	//teach_data_outputs << 1, 1, 2, 2, 1, 2, 3, 3, 1;

	double C = 10;
	int max_it = 10000; // max iterations
	double eps = 0.001; // stop algorithm when error is below this value
	int N = teach_data_inputs.rows();
	int M = teach_data_inputs.cols();

	//Starting point
	Eigen::VectorXd w0(M + 1);
	w0.setRandom();
	//w0 << 0.141886338627215, 0.421761282626275, 0.915735525189067;


	MultiSVM svm_instance("linear", true);

	t_train = static_cast<double>(timer.getTimeMicroseconds()) / 1000.0;
	svm_instance.Train(teach_data_inputs, teach_data_outputs, C, max_it, eps, w0);


	t_classify = static_cast<double>(timer.getTimeMicroseconds()) / 1000.0;
	classify_outputs = svm_instance.Classify(teach_data_inputs);


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


	MultiSVM svm_instance("quadratic", true);

	t_train = static_cast<double>(timer.getTimeMicroseconds()) / 1000.0;
	svm_instance.Train(teach_data_inputs, teach_data_outputs, C, max_it, eps);// , w0);


	t_classify = static_cast<double>(timer.getTimeMicroseconds()) / 1000.0;
	classify_outputs = svm_instance.Classify(teach_data_inputs);


	t_end = static_cast<double>(timer.getTimeMicroseconds()) / 1000.0;
#endif //MULTI_SVM_TEST_QUAD
	if (N < 15)
	{
		std::cout << "teach_data_outputs: " << teach_data_outputs << std::endl;
		std::cout << "classify_outputs: " << classify_outputs << std::endl;
	}
	else
	{
		int non_1 = 0;
		int errors = 0;
		int non_1_errors = 0;
		for (int i = 0; i < N; i++)
		{
			if (teach_data_outputs(i) != 1)
				non_1++;

			if (teach_data_outputs(i) - classify_outputs(i) != 0)
			{
				errors++;
				if (teach_data_outputs(i)!=1)
				{
					non_1_errors++;
				}
			}
		}
		std::cout << "Inputs: " << N << "\nerrors: " << errors << "\nnon-one's: " << non_1 << "\nnon-one errors: " << non_1_errors << std::endl;
	}




	std::cout << "Program runtime: " << (t_end - t_start) << " ms" << std::endl;
	std::cout << "Training runtime: " << (t_classify - t_train) << " ms" << std::endl;
	std::cout << "Classifying runtime: " << (t_end - t_classify) << " ms" << std::endl;

	return 0;
}

