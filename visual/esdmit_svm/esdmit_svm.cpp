// esdmit_svm.cpp : Defines the entry point for the console application.
//



#include "stdafx.h"
#include "Eigen/Eigen"

#include <iostream>

int _tmain(int argc, _TCHAR* argv[])
{
	Eigen::Vector2i vec(72,87);

	Eigen::Matrix<double, 9, 2> teach_data_inputs;
	Eigen::VectorXd teach_data_outputs(9);

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
	float C = 10; 
	int max_it = 10000; // max iterations
	float eps = 0.001; // stop algorithm when error is below this value
	int data_count = teach_data_outputs.size();

	//Starting point
	Eigen::VectorXd w0(data_count);
	w0.setRandom();





	std::wcout << teach_data_inputs(7, 1) << " " << teach_data_outputs(8) << " " << w0(0) << " " << w0(8) << std::endl;
	
	return 0;
}

