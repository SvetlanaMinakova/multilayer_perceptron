#pragma once
#include <iostream>
#include<ctime>
#include <vector>
#include <memory>
#include "perceptron_layer.h"
#include "loss_funcs.h"
class network
{
public:
	network(int input_len,loss_func_type loss, int batch_size=1,int max_epochs=1,double min_loss=0);
	~network(void);

	void add_layer(int _neurons_num,activation_funcs_type _act_f_type,double _act_f_type_param=0);
	void set_learning_speed(double learning_speed);
	void train(std::vector<std::shared_ptr<std::vector<double>>> &x,std::vector<std::shared_ptr<std::vector<double>>> &y);
	void network::test(std::vector<std::shared_ptr<std::vector<double>>> &x,std::vector<std::shared_ptr<std::vector<double>>> &y);
	void predict(std::shared_ptr<std::vector<double>>input);




private:
	std::vector<std::shared_ptr<perceptron_layer>> layers;
	std::shared_ptr<loss_funcs> loss;
	//characteristics
	int input_len;
	int samples_num;
	int layers_number;
	int batch_size;
	double min_loss;
	//temp data
	int max_epochs;
	int batches_in_epoch;
	double test_precision;
	double test_loss;
	double validation_loss_epoch;
	double validation_loss_batch;

	void send_signal_front(std::shared_ptr<std::vector<double>>input);
	void send_signal_back(std::shared_ptr<std::vector<double>>expected_output);
	void update_weights();

	void print_validation_batch_result(int &batch_num);
	void print_validation_epoch_result(int &epoch_num);
	void print_test_result();
	int get_batches_num_per_epoch(int &samples_num);
};

