#include "stdafx.h"
#include "BinarySVM.h"
#include <iostream>

#include <functional>

BinarySVM::BinarySVM(const std::string& aKernelFunction) : iDataCount(0), iDataDim(0)
{//todo: fix visibility of object's variables inside iKernelFunction
	BinarySVMLog("initializing binarysvm, aKernelFunction: " << aKernelFunction);
	//iKernelFunction = aKernelFunction.compare("quadratic") ? BinarySVM::LinearKernel : BinarySVM::QuadraticKernel;
	if (aKernelFunction.compare("quadratic"))
	{
		iKernelFunction = std::bind(&BinarySVM::LinearKernel, *this, std::placeholders::_1);
	}
	else
	{
		iKernelFunction = std::bind(&BinarySVM::QuadraticKernel, *this, std::placeholders::_1);
	}
}


BinarySVM::~BinarySVM()
{
}


Data_Vector_T BinarySVM::LinearKernel(const Data_Vector_T& aVector)
{
	//BinarySVMLog("LinearKernel");
	return aVector;
}

//QUAD_KERNEL_SIZE computes size of quadratic kernel where d is linear
//kernel eg.:
//-linear = [w1 w2], d = 2
// -quadratic = [w1*w1 sqrt(2)w1*w2 w2*w2] D = 3
// Note : we are using[w1 w2 b] so above example don't match
// https ://wikimedia.org/api/rest_v1/media/math/render/svg/7a8a37a96bc8bce31c758d7c378cb2b46db6ece3
//     D = 2 * d + 1; // first and last sum of the elements + c
int BinarySVM::QuadraticKernelSize(int aDimension)
{
	int D = aDimension; // first and last sum of the elements + c
	for (int i = 2; i <= aDimension; i++) // This double sum in middle
	{
		for (int j = 1; j < i; j++)
		{
			D++;
		}
	}

	return D;
}

//function[out] = quad_kernel(X_i, b)
//	% quad kernel
//	m = length(X_i);
//	D = quad_kernel_size(m);
//	out = zeros(m, 1);
//	for i = 1:m
//		out(i) = X_i(i) ^ 2;
//	end
//	for i = 2:m % This double sum in middle
//		for j = 1 : i - 1
//			out = [out; sqrt(2)*X_i(i)*X_i(j)];
//		end
//	end
//	out2 = zeros(m, 1);
//	for i = 1:m
//		out2(i) = sig_sqrt(2 * b)*X_i(i);
//	end
//	out = [out; 1]';
//end
Data_Vector_T BinarySVM::QuadraticKernel(const Data_Vector_T& aVector)
{ 
	int m = aVector.size();
	Data_Vector_T output( QuadraticKernelSize(m) );

	int k;
	for (k = 0; k < m; k++)
	{
		output(k) = aVector(k) * aVector(k);
	}
	for (int i = 1; i < m; i++)
	{
		for (int j = 0; j < i; j++)
		{
			//magic number = sqrt(2)
			output(k++) = (1.414213562373095*aVector(i)*aVector(j));
		}
	}

	return output;
}




//aTrainOutputs should have only -1's and 1's
void BinarySVM::Train(const Matrix_T& aTrainData, const Class_Vector_T& aTrainOutputs, const Data_Vector_T& aStartingVector, const float aC, const int aMaxIt, const float aEps)
{
	// SVM_GRAD gradient linear svm training algoritm for non - separable data

	//todo: erase current data (in case someone uses train() twice)
	//todo: make train() initialise its own starting vector if not given

	iDim = aStartingVector.size();
	iDataDim = aTrainData.cols();
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

	iClassificator = w;
}

//function[out] = cost_fun2(X, y, w, C, kernel)
//	% Calulates value of cost fuction to log progress or terminate algorithm
//	% http://www.robots.ox.ac.uk/~az/lectures/ml/lect2.pdf pages 29 and 36
//	[N, m] = size(X);
//	lam = 2 / (N*C);
//	regularization = (lam / 2)*w(1:end - 1)'*w(1:end-1);
//	tmp = 0;
//	for i = 1:N
//		tmp = tmp + max(1 - y(i)*(w'*kernel(X(i,:),w(end))'), 0);
//	end
//	loss_funcion = tmp / N;
//	out = regularization + loss_funcion;
//end
double BinarySVM::CostFunction(const Matrix_T& aTrainData, const Class_Vector_T& aTrainOutputs, const Data_Vector_T& aVector, const double aLam)
{
	//BinarySVMLog("aVectorsize " << aVector.size() << " iDim " << iDim);
	Data_Vector_T a = aVector.head(iDim-1);
	
	double regularization = (aLam / 2) * a.squaredNorm();
	double acc = 0;
	double tmp = 0;
	double last = aVector(iDim-1); //todo: check

	for (int i = 0; i < iDataCount; i++)
	{
		//BinarySVMLog("a.size() " << a.size() << " iKernelFunction(aTrainData.row(i)).size " << iKernelFunction(aTrainData.row(i)).size())
		tmp = 1 - aTrainOutputs(i) * (a.dot(iKernelFunction(aTrainData.row(i))) + last);
		acc += tmp > 0 ? tmp : 0;
	}
	
	return regularization + acc / iDataCount;
}


//function[out] = gradient2(X, y, w, C, kernel)
//	% http://www.robots.ox.ac.uk/~az/lectures/ml/lect2.pdf pages 36-38
//	[N, ~] = size(X);
//	m = length(w);
//	lam = 2 / (N*C);
//	L = zeros(N, m);
//	for i = 1:N
//		X_tmp = kernel(X(i, :), w(end));
//		if ((y(i)*(w'*X_tmp'))<1) % if y*f(x) >= 1 constrain is not violated so cost function = 0
//			for j = 1:m
//				L(i, j) = -y(i)*X_tmp(j);
//			end
//		end
//	end
//	out = lam*[w(1:end - 1); 0] + (sum(L)')/N;
//end
Data_Vector_T BinarySVM::Gradient(const Matrix_T& aTrainData, const Class_Vector_T& aTrainOutputs, const Data_Vector_T& aVector, const double aLam)
{
	Matrix_T L = Matrix_T::Zero(iDataCount, iDim);
	Data_Vector_T a = aVector.head(iDim-1);
	Data_Vector_T b;
	Data_Vector_T output;

	double current_output;

	for (int i = 0; i < iDataCount; i++)
	{

		b = aTrainData.row(i);
		current_output = aTrainOutputs(i);
		Data_Vector_T x_temp = iKernelFunction(b);

		//if y*f(x) >= 1 constrain is not violated so cost function = 0
		if (current_output * (a.dot( x_temp ) + aVector(iDim-1)) < 1)
		{
			for (int j = 0; j < iDim-1; j++)
			{
				L(i, j) = -current_output * x_temp(j);
			}
			L(i, iDim-1) = -current_output;
		}
	}

	output = aLam * aVector;
	output(iDim-1) = 0;
	output += L.colwise().sum() / iDataCount;

	return output;
}


Class_Vector_T BinarySVM::Classify(const Matrix_T& aData, Data_Vector_T& aProximities)
{
	int data_count = aData.rows();

	Data_Vector_T values(data_count);

	Class_Vector_T output(data_count);
	for (int i = 0; i < data_count; i++)
	{
		values(i) = iKernelFunction(aData.row(i)).dot(iClassificator.head(iDim-1)) + iClassificator(iDim-1);

		output(i) = (0 < values(i)) - (values(i) < 0); //sign() //todo: check if it exists in eigen library
	}

	//todo: consider giving aProximities only when necessary
	aProximities = values;
	return output;
}