#include "stdafx.h"
#include "BinarySVM.h"
#include <iostream>

BinarySVM::BinarySVM(const std::string& aKernelFunction) : iKernelFunction(aKernelFunction), iDataCount(0), iDim(0)
{
}


BinarySVM::~BinarySVM()
{
}

//aTrainOutputs should have only -1's and 1's
void BinarySVM::Train(const Matrix_T& aTrainData, const Class_Vector_T& aTrainOutputs, const Data_Vector_T& aStartingVector, const float aC, const int aMaxIt, const float aEps)
{
	// SVM_GRAD gradient linear svm training algoritm for non - separable data

	//todo: erase current data (in case someone uses train() twice)

	iDim = aTrainData.cols();
	iDataCount = aTrainData.rows();

	Data_Vector_T w = aStartingVector;
	Data_Vector_T w_grad;

	int iDataCount = aTrainData.rows();
	float lam = 2 / (iDataCount * aC);

	double i_cost;
	float ni;
	
	for (int it_count = 1; it_count <= aMaxIt; it_count++)
	{
		ni = 1 / (lam * it_count);
		i_cost = CostFunction(aTrainData, aTrainOutputs, w, lam);

		if (i_cost <= aEps)
		{
			BinarySVMLog("Breaking after " << it_count << " iterations.");
			break;
		}

		w_grad = Gradient(aTrainData, aTrainOutputs, w, lam);
		w -= ni*w_grad;
	}

	iVector = w;
}


double BinarySVM::CostFunction(const Matrix_T& aTrainData, const Class_Vector_T& aTrainOutputs, const Data_Vector_T& aVector, const float aLam)
{
	// Calulates value of cost fuction to log progress or terminate algorithm
	// http://www.robots.ox.ac.uk/~az/lectures/ml/lect2.pdf pages 29 and 36

	Data_Vector_T a = aVector.head(iDim);
	Data_Vector_T b;

	double regularization = (aLam / 2) * a.squaredNorm();
	double acc = 0;
	double tmp = 0;
	double last = aVector(iDim);

	Data_Vector_T temp_vec = aTrainData * a;
	for (int i = 0; i < iDataCount; i++)
	{
		tmp = 1 - aTrainOutputs(i) * (temp_vec(i) + last);
		acc += tmp > 0 ? tmp : 0;
	}
	
	return regularization + acc / iDataCount;
}


Data_Vector_T BinarySVM::Gradient(const Matrix_T& aTrainData, const Class_Vector_T& aTrainOutputs, const Data_Vector_T& aVector, const float aLam)
{
	// http://www.robots.ox.ac.uk/~az/lectures/ml/lect2.pdf pages 36-38
	//BinarySVMLog("Gradient() entering");

	Matrix_T L = Matrix_T::Zero(iDataCount, iDim + 1);
	Data_Vector_T a = aVector.head(iDim);
	Data_Vector_T b;
	Data_Vector_T output;

	double current_output;

	for (int i = 0; i < iDataCount; i++)
	{
		b = aTrainData.row(i);
		current_output = aTrainOutputs(i);
		
		//if y*f(x) >= 1 constrain is not violated so cost function = 0
		if (current_output * (a.dot(b) + aVector(iDim)) < 1)
		{
			for (int j = 0; j < iDim; j++)
			{
				L(i, j) = -current_output * aTrainData(i, j);
			}
			L(i, iDim) = -current_output;
		}
	}

	output = aLam * aVector;
	output(iDim) = 0;
	output += L.colwise().sum() / iDataCount;

	return output;
}


Class_Vector_T BinarySVM::Classify(const Matrix_T& aData, Data_Vector_T& aProximities)
{
	int data_count = aData.rows();

	Data_Vector_T values = aData * iVector.head(iDim) + iVector(iDim) * Data_Vector_T::Ones(data_count);

	Class_Vector_T output(data_count);
	for (int i = 0; i < data_count; i++)
	{
		output(i) = (0 < values(i)) - (values(i) < 0); //sign()
	}
	
	
	aProximities = values;
	return output;
}