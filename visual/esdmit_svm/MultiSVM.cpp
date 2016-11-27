#include "stdafx.h"
#include "MultiSVM.h"


MultiSVM::MultiSVM(const std::string& aKernelFunction)
{
}


MultiSVM::~MultiSVM()
{
}


void MultiSVM::Train(const Matrix_T& aTrainData, const Class_Vector_T& aTrainOutputs, const Data_Vector_T& aStartingVector, const float aC, const int aMaxIt, const float aEps)
{
	//initialize data and parameters


	//find unique classes (or their count)


	//create output vectors 1 vs the rest


	//teach binary svm for every output vector


}


Class_Vector_T MultiSVM::Classify(const Matrix_T& aData)
{
	//test each svm


	//for each row choose the best

	return Class_Vector_T();
}