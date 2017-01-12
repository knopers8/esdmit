#pragma once

#include "BinarySVM.h"
#include <vector>

#define TRAIN_THREADING

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

	static void TrainingThread(BinarySVM * aSVM, const Matrix_T& aTrainData, const Class_Vector_T aTrainOutputs, const Data_Vector_T aStartingVector, const float aC, const int aMaxIt, const float aEps);

public:

	MultiSVM(const std::string& aKernelFunction = "linear", bool aNormalize = false);
	
	~MultiSVM();

	//Matrix_T Normalize(const Matrix_T& aData);

	static int QuadraticKernelSize(int aDimension);

	bool Train(const Matrix_T& aTrainData, const Class_Vector_T& aTrainOutputs, const float aC = 10, const int aMaxIt = 10000, const float aEps = 0.001, const Data_Vector_T& aStartingVector = Data_Vector_T());

	Class_Vector_T Classify(const Matrix_T& aData);

};

