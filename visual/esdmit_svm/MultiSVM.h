#pragma once

#include "BinarySVM.h"
#include <vector>

class MultiSVM
{
private:

	int iDim;

	int iDataCount;

	int iClassesCount;

	std::string iKernelFunction;

	std::vector<BinarySVM> iSVMList;

public:

	MultiSVM(const std::string& aKernelFunction = "linear");
	
	~MultiSVM();

	void Train(const Matrix_T& aTrainData, const Class_Vector_T& aTrainOutputs, const Data_Vector_T& aStartingVector, const float aC, const int aMaxIt, const float aEps);

	Class_Vector_T Classify(const Matrix_T& aData);

};

