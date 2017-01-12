#pragma once

#include "Eigen/Eigen"
#include <string>


#ifndef Matrix_T
#define Matrix_T Eigen::MatrixXd
#endif

#ifndef Data_Vector_T
#define Data_Vector_T Eigen::VectorXd
#endif

#ifndef Class_Vector_T
#define Class_Vector_T Eigen::VectorXi
#endif


#define BinarySVMLog( a ) { std::cout << a << std::endl; }


class BinarySVM
{
private:

	int iDim;

	int iDataDim;

	int iDataCount;

	Data_Vector_T iClassificator;

	std::string iKernelType;
//public:
	std::function < Data_Vector_T(const Data_Vector_T&) > iKernelFunction;

	Data_Vector_T LinearKernel(const Data_Vector_T& aVector);

	Data_Vector_T QuadraticKernel(const Data_Vector_T& aVector);

	double CostFunction(const Matrix_T& aTrainData, const Class_Vector_T& aTrainOutputs, const Data_Vector_T& aVector, const double aLam);

	Data_Vector_T Gradient(const Matrix_T& aTrainData, const Class_Vector_T& aTrainOutputs, const Data_Vector_T& aVector, const double aLam);

public:
	
	BinarySVM(const std::string& aKernelFunction = "linear");

	BinarySVM(const BinarySVM & aBinarySVM);

	~BinarySVM();

	static int QuadraticKernelSize(int aDimension);

	void Train(const Matrix_T& aTrainData, const Class_Vector_T& aTrainOutputs,  const Data_Vector_T& aStartingVector, const float aC, const int aMaxIt, const float aEps);

	Class_Vector_T Classify(const Matrix_T& aData, Data_Vector_T& aProximities);

	void Normalize( Matrix_T& aData ); //todo

};

