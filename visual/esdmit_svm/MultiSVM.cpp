#include "stdafx.h"
#include "MultiSVM.h"
#include <iostream>


MultiSVM::MultiSVM(const std::string& aKernelFunction, bool aNormalize) : iKernelType(aKernelFunction), iNormalize(aNormalize), iDataCount(0), iDataDim(0)
{
}


MultiSVM::~MultiSVM()
{
}

int MultiSVM::QuadraticKernelSize(int aDimension)
{
	return BinarySVM::QuadraticKernelSize(aDimension);
}

Matrix_T MultiSVM::NormalizeTrainData(const Matrix_T& aData)
{
	Matrix_T normalized(iDataCount, iDataDim);

	for (int j = 0; j < iDataDim; j++)
	{
		Data_Vector_T vec = aData.col(j);
		double mean = vec.mean();
		double std = sqrt((vec - mean * Data_Vector_T::Ones(iDataCount)).squaredNorm() / (iDataCount-1));

		normalized.col(j) = (vec - mean * Data_Vector_T::Ones(iDataCount)) / std;
		iMeans.push_back(mean);
		iStds.push_back(std);
	}
	
	return normalized;
}

Matrix_T MultiSVM::NormalizeClassifyData(const Matrix_T& aData)
{
	Matrix_T normalized(iDataCount, iDataDim);

	for (int j = 0; j < iDataDim; j++)
	{
		normalized.col(j) = (aData.col(j) - iMeans[j] * Data_Vector_T::Ones(iDataCount)) / iStds[j];
	}

	return normalized;
}



//aTrainOutputs classes should be numbers from 1 to x, but not e.g. 1,2,4
void MultiSVM::Train(const Matrix_T& aTrainData, const Class_Vector_T& aTrainOutputs, const float aC, const int aMaxIt, const float aEps, const Data_Vector_T& aStartingVector)
{
	//todo: when there are only two classes, use one binary svm
	
	//find unique classes (or their count) //todo: gather classes id properly
	iClassesCount = 0;
	std::cout << "aTrainOutputs.size() " << aTrainOutputs.size() << std::endl;
	for (int i = 0; i < aTrainOutputs.size(); i++)
	{
		if (iClassesCount < aTrainOutputs(i))
		{
			std::cout << aTrainOutputs(i) << std::endl;
			iClassesCount++;
		}
	}
	std::cout << "iClassesCount: " << iClassesCount << std::endl;
	
	//initialize data and parameters
	iSVMList = std::vector<BinarySVM>(iClassesCount, BinarySVM(iKernelType));
	iDataDim = aTrainData.cols();
	iDataCount = aTrainData.rows();
	Matrix_T train_data;

	//initialize starting vector
	Class_Vector_T binary_outputs(iDataCount);
	Data_Vector_T starting_vector;
	if (aStartingVector.isZero())
	{
		starting_vector = iKernelType.compare("quadratic") ? Data_Vector_T(iDataDim + 1) : Data_Vector_T(QuadraticKernelSize(iDataDim) + 1);
		starting_vector.setRandom();
	}
	else
	{
		starting_vector = aStartingVector;
	}


	//normalize
	train_data = iNormalize ? NormalizeTrainData(aTrainData) : aTrainData;

	//teach each binary svm	
	for (int class_id = 1; class_id <= iClassesCount; class_id++)
	{
		std::cout << "teaching class_ID: " << class_id << std::endl;
		
		//create output vectors 1 vs the rest
		for (int j = 0; j < iDataCount; j++)
		{
			binary_outputs(j) = aTrainOutputs(j) == class_id ? 1 : -1;
		}

		iSVMList[class_id - 1].Train(train_data, binary_outputs, starting_vector, aC, aMaxIt, aEps);
	}
}


Class_Vector_T MultiSVM::Classify(const Matrix_T& aData)
{
	Matrix_T train_data;

	//normalize data if necessary
	train_data = iNormalize ? NormalizeClassifyData(aData) : aData;

	//test each svm
	std::vector<Data_Vector_T> proximity_results(iClassesCount);
	for (int i = 0; i < iClassesCount; i++)
	{
		iSVMList[i].Classify(train_data, proximity_results[i]);
	}

	//for each row choose the best 
	int data_count = train_data.rows();
	Class_Vector_T results(data_count);
	for (int i = 0; i < data_count; i++)
	{
		results(i) = 0;
		double best = -100000000;
		for (int j = 0; j < iClassesCount; j++)
		{
			if (proximity_results[j](i) > best)
			{
				best = proximity_results[j](i);
				results(i) = j + 1;
			}
		}
	}
	return results;
}