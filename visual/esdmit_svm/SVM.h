#pragma once

#include "Eigen\Eigen"
#include <string>




class SVM
{
private:

	std::string iKernelFunction;

public:

	SVM();

	SVM( const std::string& aKernelFunction );

	~SVM();

	bool Train( const Eigen::Matrix2Xf& aTrainData );

	bool Normalize( Eigen::Matrix2Xf& aData );

};

