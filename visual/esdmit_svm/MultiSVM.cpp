#include "stdafx.h"
#include "MultiSVM.h"

#include <iostream>
#include <algorithm>
#include <thread>


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


void MultiSVM::TrainingThread(BinarySVM * aSVM, const Matrix_T& aTrainData, const Class_Vector_T aTrainOutputs, const Data_Vector_T aStartingVector, const float aC, const int aMaxIt, const float aEps)
{
	aSVM->Train(aTrainData, aTrainOutputs, aStartingVector, aC, aMaxIt, aEps);
	std::cout << "Training - thread ended." << std::endl;
}


void MultiSVM::Train(const Matrix_T& aTrainData, const Class_Vector_T& aTrainOutputs, const float aC, const int aMaxIt, const float aEps, const Data_Vector_T& aStartingVector)
{
	//find unique classes (or their count)
	std::cout << "aTrainOutputs.size() " << aTrainOutputs.size() << std::endl;
	for (int i = 0; i < aTrainOutputs.size(); i++)
	{
		if (std::find(iClassesList.begin(), iClassesList.end(), aTrainOutputs(i)) == iClassesList.end())
		{
			std::cout << aTrainOutputs(i) << std::endl;
			iClassesList.push_back(aTrainOutputs(i));
		}
	}
	std::cout << "iClassesList.size(): " << iClassesList.size() << std::endl;

	if (iClassesList.size() < 2)
	{
		std::cout << "iClassesList.size() < 2, returning" << std::endl;
		return;
	}
		
	
	//initialize data and parameters
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

	if (iClassesList.size() == 2)
	{
		iSVMList = std::vector<BinarySVM>(1, BinarySVM(iKernelType));

		for (int j = 0; j < iDataCount; j++)
		{
			binary_outputs(j) = aTrainOutputs(j) == iClassesList[1] ? 1 : -1;
		}
		iSVMList[0].Train(train_data, binary_outputs, starting_vector, aC, aMaxIt, aEps);
	}
	else 
	{
		iSVMList = std::vector<BinarySVM>(iClassesList.size(), BinarySVM(iKernelType));

#ifdef TRAIN_THREADING
		std::vector<std::thread> thread_list(iClassesList.size());
#endif 

		//teach each binary svm	
		for (unsigned int class_id = 0; class_id < iClassesList.size(); class_id++)
		{
			//create output vectors 1 vs the rest
			for (int j = 0; j < iDataCount; j++)
			{
				binary_outputs(j) = aTrainOutputs(j) == iClassesList[class_id] ? 1 : -1;
			}

#ifdef TRAIN_THREADING
			std::cout << "Training - thread for class " << iClassesList[class_id] << " started." << std::endl;
			thread_list[class_id] = std::thread(&MultiSVM::TrainingThread, &iSVMList[class_id], train_data, binary_outputs, starting_vector, aC, aMaxIt, aEps);
#else 
			std::cout << "teaching class_ID: " << iClassesList[class_id] << std::endl;
			iSVMList[class_id].Train(train_data, binary_outputs, starting_vector, aC, aMaxIt, aEps);
#endif 
		}

#ifdef TRAIN_THREADING
		for (auto && thr : thread_list)
		{
			thr.join();
		}
#endif

	}
	std::cout << "Training complete" << std::endl;
}


Class_Vector_T MultiSVM::Classify(const Matrix_T& aData)
{
	Matrix_T classify_data;

	//normalize data if necessary
	classify_data = iNormalize ? NormalizeClassifyData(aData) : aData;

	if (iClassesList.size() < 2)
	{
		return Class_Vector_T(aData.rows());
	}
	else if (iClassesList.size() == 2)
	{
		Data_Vector_T proximity_results;

		Class_Vector_T results = iSVMList[0].Classify(classify_data, proximity_results);

		for (int i = 0; i < results.size(); i++)
		{
			results(i) = results(i) == 1 ? iClassesList[1] : iClassesList[0];
		}

		return results;
	}
	else
	{
		//test each svm
		std::vector<Data_Vector_T> proximity_results(iClassesList.size());
		for (unsigned int i = 0; i < iClassesList.size(); i++)
		{
			std::cout << "Classifying " << iClassesList[i] << " class" << std::endl;
			iSVMList[i].Classify(classify_data, proximity_results[i]);
		}

		//for each row choose the best 
		int data_count = classify_data.rows();
		Class_Vector_T results(data_count);
		for (int i = 0; i < data_count; i++)
		{
			results(i) = 0;
			double best = -100000000;
			for (unsigned int j = 0; j < iClassesList.size(); j++)
			{
				if (proximity_results[j](i) > best)
				{
					best = proximity_results[j](i);
					results(i) = iClassesList[j];
				}
			}
		}
		return results;
	}
}