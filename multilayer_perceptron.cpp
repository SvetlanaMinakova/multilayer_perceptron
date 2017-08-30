#include "stdafx.h"

#include <vector>
#include <iostream>
#include <memory>
#include "network.h" 
#include "input_data_manager.h"

int _tmain(int argc, _TCHAR* argv[])
{//input data configuration
	//load mock data from text files
	// it is considered that every file represents new data class
	auto source_data_samples = input_data_manager::get_all_data_from_InputTxtData_folder();
	int source_data_samples_num = source_data_samples.size();
	int inp_len = input_data_manager::get_input_data_len(source_data_samples);
	int outp_len=source_data_samples_num;
	auto source_data_labels=input_data_manager::get_binary_labels(source_data_samples_num);

	//create training data
	int train_samples_num =256;
	double train_samples_noise_level=0.1f;
	std::vector<std::shared_ptr<std::vector<double>>> train_x;
	std::vector<std::shared_ptr<std::vector<double>>> train_y;
	train_x.reserve(train_samples_num);
	train_y.reserve(train_samples_num);
	input_data_manager::create_samples(source_data_samples,source_data_labels,train_x,train_y,inp_len,train_samples_num,train_samples_noise_level);

	//create testing data
	int test_samples_num =100;
	double test_samples_noise_level=0.1f;
	std::vector<std::shared_ptr<std::vector<double>>> test_x;
	std::vector<std::shared_ptr<std::vector<double>>> test_y;
	train_x.reserve(test_samples_num);
	train_y.reserve(test_samples_num);
	input_data_manager::create_samples(source_data_samples,source_data_labels,test_x,test_y,inp_len,train_samples_num,train_samples_noise_level);

	//create network
	int batch_size=32;
	int max_epochs=10;
	double min_loss=0.01f;

	network test_network=network(inp_len,loss_func_type::CROSSENTROPY,batch_size,max_epochs,min_loss);
	test_network.add_layer(10,activation_funcs_type::SIGM);
	test_network.add_layer(outp_len,activation_funcs_type::SOFTMAX);

	test_network.train(train_x,train_y);
	test_network.test(test_x,test_y);

	system("pause");
	return 0;
}