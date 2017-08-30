#pragma once
#include <vector>
#include <memory>
#include "matrix_and_vectors_operations.h"
#include "activation_funcs_worker.h"

class perceptron_layer
{
public:
	perceptron_layer::perceptron_layer(int &inputs_num,int &neurons_num, activation_funcs_type act_f_type, double act_f_type_param);
	perceptron_layer::perceptron_layer(std::shared_ptr<perceptron_layer> prev_layer_ptr,
		int &neurons_num,activation_funcs_type act_f_type, double act_f_type_param);
	~perceptron_layer(void);

	std::shared_ptr<perceptron_layer> prev_layer_ptr;
	std::shared_ptr<perceptron_layer> next_layer_ptr;
	std::shared_ptr<activation_funcs_worker> act_f;
	
	//intermediate data containers
	std::shared_ptr<std::vector<double>> input;
	std::vector<std::vector<double>>weights;
	std::vector<double>non_activated_stages;
	//error for one sample
	std::vector<double>error;
	//summary error for batch or mini-batch
	std::vector<double>partly_summ_error;
	std::shared_ptr<std::vector<double>>output;
	std::shared_ptr<std::vector<double>>gradient_from_prev_layer;

	int outputs_number;
    int inputs_number;
	int neurons_number;
	double learning_speed;

	void connect_input(std::shared_ptr<std::vector<double>> _input);
	void connect_next_layer(std::shared_ptr<perceptron_layer> _next_layer_ptr);
	void set_learning_speed(double speed);
	
	void get_output();
	//for output layer
	void get_error(std::shared_ptr<std::vector<double>> expected_output);
	//for hidden layers
	void get_error();
	void update_weights();
	//
	void print_input();
	void print_output();
    void print_error();
	void print_weights();

    private:
	void init_intermediate_data_containers(activation_funcs_type act_f_type, double &act_f_type_param);
	void update_partly_summ_error();
	void clear_partly_summ_error();
	void count_neg_gradient_for_prev_layer(std::shared_ptr<std::vector<double>> gradient_placeholder);
};