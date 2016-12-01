// esdmit_svm.cpp : Defines the entry point for the console application.
//



#include "stdafx.h"
#include "Eigen/Eigen"
#include "BinarySVM.h"
#include "MultiSVM.h"

#include <iostream>

#define BINARY_SVM_TESTx
#define MULTI_SVM_TEST

int _tmain(int argc, _TCHAR* argv[])
{
#ifdef BINARY_SVM_TEST
	Eigen::Matrix<double, 9, 2> teach_data_inputs;
	Eigen::VectorXi teach_data_outputs(9);
	Eigen::VectorXi classify_outputs;

	teach_data_inputs << 1, 3,
						 2, 2,
						 3, 1,
						 3, 2,
						 6, 1,
						 5, 2,
						 4, 3,
						 3, 4,
						 1, 1;
	
	teach_data_outputs << 1, 1, 1, 1, -1, -1, -1, -1, -1 ;


	// Paramteter C(0, inf) small C = fuck contraints, large margin big.C = use constrains, smaller margin but less mistakes in training data
	// C = inf enforces no mistakes in training data, not always possible so
	// output may be some random shit
	double C = 10; 
	int max_it = 10000; // max iterations
	double eps = 0.001; // stop algorithm when error is below this value
	int N = teach_data_inputs.rows();
	int M = teach_data_inputs.cols();

	//Starting point
	Eigen::VectorXd w0(M+1);
	w0.setRandom();
	//w0 << 0.141886338627215, 0.421761282626275, 0.915735525189067;


	BinarySVM svm_instance;

	svm_instance.Train(teach_data_inputs, teach_data_outputs, w0, C, max_it, eps);
	

	classify_outputs = svm_instance.Classify(teach_data_inputs);
	std::cout << classify_outputs << std::endl;
#endif // BINARY_SVM_TEST
#ifdef MULTI_SVM_TEST
	Eigen::Matrix<double, 9, 2> teach_data_inputs;
	Eigen::VectorXi teach_data_outputs(9);
	Eigen::VectorXi classify_outputs;

	teach_data_inputs << 1, 3,
						2, 2,
						7, 7,
						7, 9,
						6, 1,
						8, 6,
						-7, -7,
						-7, -9,
						1, 1;

	teach_data_outputs << 1, 1, 2, 2, 1, 2, 3, 3, 1;

	double C = 10;
	int max_it = 10000; // max iterations
	double eps = 0.001; // stop algorithm when error is below this value
	int N = teach_data_inputs.rows();
	int M = teach_data_inputs.cols();

	//Starting point
	Eigen::VectorXd w0(M + 1);
	w0.setRandom();
	//w0 << 0.141886338627215, 0.421761282626275, 0.915735525189067;


	MultiSVM svm_instance;

	svm_instance.Train(teach_data_inputs, teach_data_outputs, w0, C, max_it, eps);

	classify_outputs = svm_instance.Classify(teach_data_inputs);

	std::cout << "teach_data_outputs: " << teach_data_outputs << std::endl;
	std::cout << "classify_outputs: " << classify_outputs << std::endl;

#endif //MULTI_SVM_TEST

	return 0;
}

