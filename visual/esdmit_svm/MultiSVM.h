#pragma once

#include "BinarySVM.h"
#include <vector>

class MultiSVM
{
private:
	
	bool iNormalize;

	int iDataDim;

	int iDataCount;

	std::vector<double> iMeans;

	std::vector<double> iStds;

	std::string iKernelType;

	std::vector<BinarySVM> iSVMList;

	std::vector<int> iClassesList;

public:

	Matrix_T NormalizeTrainData(const Matrix_T& aData);

	Matrix_T NormalizeClassifyData(const Matrix_T& aData);

public:

	MultiSVM(const std::string& aKernelFunction = "linear", bool aNormalize = false);
	
	~MultiSVM();

	//Matrix_T Normalize(const Matrix_T& aData);

	static int QuadraticKernelSize(int aDimension);

	void Train(const Matrix_T& aTrainData, const Class_Vector_T& aTrainOutputs, const float aC, const int aMaxIt, const float aEps, const Data_Vector_T& aStartingVector = Data_Vector_T());

	Class_Vector_T Classify(const Matrix_T& aData);

};

