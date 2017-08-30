#include "stdafx.h"
#include "perceptron_layer.h"


perceptron_layer::perceptron_layer(int &_inputs_num,int &_neurons_num,
								   activation_funcs_type _act_f_type,double _act_f_type_param=0)
{
	//init parameters
	prev_layer_ptr=nullptr;
	learning_speed=0.1f;
	inputs_number = _inputs_num;
	neurons_number = _neurons_num;
	outputs_number = _neurons_num;
	init_intermediate_data_containers(_act_f_type,_act_f_type_param);
}

perceptron_layer::perceptron_layer(std::shared_ptr<perceptron_layer> _prev_layer_ptr,int &_neurons_num,
								   activation_funcs_type _act_f_type,double _act_f_type_param=0)
{
	//init parameters
	prev_layer_ptr=_prev_layer_ptr;
	learning_speed=0.1f;
	inputs_number = _prev_layer_ptr->outputs_number;
	input = _prev_layer_ptr->output;
	neurons_number = _neurons_num;
	outputs_number = _neurons_num;
	init_intermediate_data_containers(_act_f_type,_act_f_type_param);
	
	prev_layer_ptr=_prev_layer_ptr;
}

perceptron_layer::~perceptron_layer(void)
{
}

void perceptron_layer::connect_input(std::shared_ptr<std::vector<double>> _input)
{input=_input;}

void perceptron_layer::connect_next_layer(std::shared_ptr<perceptron_layer> _next_layer_ptr)
{next_layer_ptr=_next_layer_ptr;}

void perceptron_layer::set_learning_speed(double new_l_speed)
{learning_speed=new_l_speed;}

void perceptron_layer::init_intermediate_data_containers(activation_funcs_type act_f_type, double &act_f_type_param)
{	
	//init activation func
	if(act_f_type_param==0)
		act_f=activation_funcs_worker::create_activation_funcs_worker(act_f_type);
	else
		act_f=activation_funcs_worker::create_activation_funcs_worker(act_f_type,act_f_type_param);
	//init all vector structures with zeros
	non_activated_stages.resize(neurons_number);

	error.resize(outputs_number);
	partly_summ_error.resize(outputs_number);
	//temp for keeping partly_summ_error for one sample
	gradient_from_prev_layer=std::make_shared<std::vector<double>>();
	gradient_from_prev_layer->resize(outputs_number);

	output=std::make_shared<std::vector<double>>();
	output->resize(outputs_number);

	weights.resize(inputs_number);
	for (int i=0; i<inputs_number;++i)
		{
			weights[i].resize(outputs_number);
		}

	//init weights with init values depends on activation_func type
	double scale_factor=act_f->get_scale_factor_for_weights_init(neurons_number);
	matrix_and_vectors_operations::init_matrix_random(weights,scale_factor);
}


void perceptron_layer::get_output()
{	
	for(int j=0; j<neurons_number;j++)
	{
		non_activated_stages[j]=0;
	    //set new value
		for(int i=0;i<inputs_number;i++)
		{
			non_activated_stages[j]+=input->at(i)*weights[i][j];
		}
	}
	act_f->apply(non_activated_stages,*(output));
}

void perceptron_layer::get_error(std::shared_ptr<std::vector<double>> expected_output)
{
	for(int i=0; i<outputs_number;i++)
		error[i]=(output->at(i)-expected_output->at(i));

	update_partly_summ_error();
}

void perceptron_layer::get_error()
{ //err = f_derived(ul)  * neg_gradient_next_layer (all operations are element-wise) 
	
	act_f->apply_d(non_activated_stages,error);
	next_layer_ptr->count_neg_gradient_for_prev_layer(gradient_from_prev_layer);

	for(int j=0; j<outputs_number;j++)
	{ 
		error[j] *= gradient_from_prev_layer->at(j);
	}
	update_partly_summ_error();
}

 //neg_gradient_next_layer[l] = sigma_next_layer[l-1] * W_transponed[l-1] (all operations are element-wise) 
void perceptron_layer::count_neg_gradient_for_prev_layer(std::shared_ptr<std::vector<double>> gradient_placeholder)
{
	double part_summ_error;
	for(int j=0; j<outputs_number;j++)
	{ 
		part_summ_error =0;
		for(int i=0; i<neurons_number;i++)
		{
			part_summ_error+=error[i]*weights[j][i];
		}	
		gradient_placeholder->at(j) = part_summ_error;
   }
}

void perceptron_layer::update_partly_summ_error()
{
	for(int j=0;j<outputs_number;j++)
		partly_summ_error[j]+=error[j];
}

void perceptron_layer::update_weights()
{
	for(int j=0; j<outputs_number;j++)
	{
		for(int i=0; i<inputs_number;i++)
			weights[i][j]-=partly_summ_error[j] * input->at(i) * learning_speed;
	}
	clear_partly_summ_error();
}

void perceptron_layer::clear_partly_summ_error()
{
	matrix_and_vectors_operations::set_zero_values(partly_summ_error);
}

void perceptron_layer::print_input()
{matrix_and_vectors_operations::print_vector(input);}

void perceptron_layer::print_output()
{matrix_and_vectors_operations::print_vector(output);}

void perceptron_layer::print_error()
{matrix_and_vectors_operations::print_vector(error);}

void perceptron_layer::print_weights()
{matrix_and_vectors_operations::print_matrix(weights);}



