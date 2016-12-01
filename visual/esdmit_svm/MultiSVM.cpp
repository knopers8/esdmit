#include "stdafx.h"
#include "MultiSVM.h"
#include <iostream>


MultiSVM::MultiSVM(const std::string& aKernelFunction)
{
}


MultiSVM::~MultiSVM()
{
}

//aTrainOutputs classes should be numbers from 1 to x, but not e.g. 1,2,4
void MultiSVM::Train(const Matrix_T& aTrainData, const Class_Vector_T& aTrainOutputs, const Data_Vector_T& aStartingVector, const float aC, const int aMaxIt, const float aEps)
{
	//find unique classes (or their count)
	iClassesCount = 0;
	for (int i = 0; i < aTrainOutputs.size(); i++)
	{
		if (iClassesCount < aTrainOutputs(i))
		{
			iClassesCount++;
		}
	}
	//std::cout << "iClassesCount: " << iClassesCount << std::endl;
	//initialize data and parameters
	iSVMList = std::vector<BinarySVM>(iClassesCount);
	iDim = aTrainData.cols();
	iDataCount = aTrainData.rows();

	
	Class_Vector_T binary_outputs(iDataCount);
	Data_Vector_T starting_vector(iDim + 1);
	starting_vector.setRandom();

	for (int class_id = 1; class_id <= iClassesCount; class_id++)
	{
		//create output vectors 1 vs the rest
		for (int j = 0; j < iDataCount; j++)
		{
			binary_outputs(j) = aTrainOutputs(j) == class_id ? 1 : -1;
		}

		//std::cout << "class_ID: " << class_id << std::endl << binary_outputs << std::endl;
		//teach binary svm for every output vector
		iSVMList[class_id - 1].Train(aTrainData, binary_outputs, starting_vector, aC, aMaxIt, aEps);
	}
	
}


Class_Vector_T MultiSVM::Classify(const Matrix_T& aData)
{
	//test each svm
	std::vector<Class_Vector_T> binary_results(iClassesCount);
	for (int i = 0; i < iClassesCount; i++)
	{
		binary_results[i] = iSVMList[i].Classify(aData);
		//std::cout << "binary_results " << i << " " << binary_results[i] << std::endl;
	}

	//for each row choose the best //todo: use distance instead, when it will be available
	int data_count = aData.rows();
	Class_Vector_T results(data_count);
	for (int i = 0; i < data_count; i++)
	{
		results(i) = 0;
		for (int j = 0; j < iClassesCount; j++)
		{
			if (binary_results[j](i) == 1)
			{
				results(i) = j + 1;
				break;
			}
		}
	}


	return results;
}