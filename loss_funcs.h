#pragma once
#include <vector>
#include <algorithm>
#include "matrix_and_vectors_operations.h"
#include <memory>

enum loss_func_type {MSE,CROSSENTROPY}; 


class loss_funcs{
public:
	loss_funcs(void);
	virtual ~loss_funcs(void);
	virtual double count_loss(std::shared_ptr<std::vector<double>> output, std::shared_ptr<std::vector<double>> expected_output)=0;
	//virtual void count_gradient (std::vector<double> &output, std::vector<double> &expected_output, std::vector<double> &result);
	static std::shared_ptr<loss_funcs> create_loss (loss_func_type lf_type);
	//for batch and mini-batch gradient descent
	//static int batch_counter;
};

class MSE_loss: public loss_funcs
{
public:
	double count_loss(std::shared_ptr<std::vector<double>> output, std::shared_ptr<std::vector<double>> expected_output)
	{
		double loss=0;
		double diff;
		for (int id=0; id<output->size();id++)
		{
			diff=output->at(id)-expected_output->at(id);
			loss+=diff*diff;
		}
		loss=loss/2;
		return loss;
	}
};

class CROSSENTROPY_loss: public loss_funcs
{
public:
	double count_loss(std::shared_ptr<std::vector<double>> output, std::shared_ptr<std::vector<double>> expected_output)
	{
		double loss=0;
		for (int id=0; id<output->size();id++)
		{
			if(output->at(id)!=0)
				loss+=expected_output->at(id)*std::log(output->at(id));
		}
		return loss*(-1);
	}
};