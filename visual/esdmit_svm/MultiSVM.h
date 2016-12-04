#pragma once

#include "BinarySVM.h"
#include <vector>

class MultiSVM
{
private:
	
	bool iNormalize;

	int iDim;

	int iDataCount;

	int iClassesCount;

	std::vector<double> iMeans;

	std::vector<double> iStds;

	std::string iKernelFunction;

	std::vector<BinarySVM> iSVMList;

	Matrix_T NormalizeTrainData(const Matrix_T& aData);

	Matrix_T NormalizeClassifyData(const Matrix_T& aData);

public:

	MultiSVM(const std::string& aKernelFunction = "linear", bool aNormalize = false);
	
	~MultiSVM();

	//Matrix_T Normalize(const Matrix_T& aData);

	void Train(const Matrix_T& aTrainData, const Class_Vector_T& aTrainOutputs, const Data_Vector_T& aStartingVector, const float aC, const int aMaxIt, const float aEps);

	Class_Vector_T Classify(const Matrix_T& aData);

};

