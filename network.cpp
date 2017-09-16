#include "stdafx.h"
#include "network.h"



network::network(int _input_len,loss_func_type _loss, int _batch_size,int _max_epochs,double _min_loss)
{ 
	input_len=_input_len;
	batch_size = _batch_size;
	max_epochs=_max_epochs;
	min_loss= _min_loss;
	loss=loss_funcs::create_loss(_loss);
	layers_number=0;
}

network::~network(void)
{
}

void network::add_layer(int _neurons_num,activation_funcs_type _act_f_type,double _act_f_type_param)
{
	std::shared_ptr<perceptron_layer> new_layer;
	if(layers_number==0)
		{
			new_layer = std::make_shared<perceptron_layer>(input_len,_neurons_num,_act_f_type,_act_f_type_param);
		}
	else
		{
			new_layer = std::make_shared<perceptron_layer>(layers[layers_number-1],_neurons_num,_act_f_type,_act_f_type_param);
			layers[layers_number-1]->connect_next_layer(new_layer);
		}
	layers.push_back(new_layer);
	layers_number++;
}

void network::set_learning_speed(double _learning_speed)
{
	for(auto layer: layers)
		layer->set_learning_speed(_learning_speed);
}

void network::train(std::vector<std::shared_ptr<std::vector<double>>> &x,std::vector<std::shared_ptr<std::vector<double>>> &y)
{ 
  srand(42);
  samples_num = x.size();
  batches_in_epoch=get_batches_num_per_epoch(samples_num);
  validation_loss_epoch=0;
  validation_loss_batch=0;
  int cur_sample_id=0;
  int cur_epoch=0;
  //
  while (cur_epoch<max_epochs)
	{ 
		for(int bn=0; bn<batches_in_epoch;++bn) 
		{
			for(int cur_sample_in_batch=0; cur_sample_in_batch<batch_size;++cur_sample_in_batch)
				{
					cur_sample_id =(int)(rand()%samples_num);
					send_signal_front(x[cur_sample_id]);
					send_signal_back(y[cur_sample_id]);

					validation_loss_batch+=loss->count_loss(layers[layers_number-1]->output,y[cur_sample_id]);
				}
			update_weights();

			validation_loss_batch/=(double)(batch_size);
			validation_loss_epoch+=validation_loss_batch;
			print_validation_batch_result(bn);
			validation_loss_batch=0;
		}
			validation_loss_epoch/=double(batches_in_epoch);
			print_validation_epoch_result(cur_epoch);

			//early-stopping
			if(validation_loss_epoch<min_loss)
				break;

			validation_loss_epoch=0;
			cur_epoch++;
			
	}
	

}



void network::test(std::vector<std::shared_ptr<std::vector<double>>> &x,std::vector<std::shared_ptr<std::vector<double>>> &y)
{  
	int cur_sample_id=0;
	int test_data_size=x.size();
	int errors_number=0;
	test_loss=0;

	for(int test_iter=0; test_iter<test_data_size; test_iter++)
	 {
		 cur_sample_id=rand()%test_data_size;
		 send_signal_front(x[cur_sample_id]);
		 //count partly loss, precision
		 if(matrix_and_vectors_operations::classes_not_match(layers[layers_number-1]->output,y[cur_sample_id]))
			 errors_number++;
		 test_loss+=loss->count_loss(layers[layers_number-1]->output,y[cur_sample_id]);                 
	 }
	test_precision=((double)(test_data_size-errors_number)/(double)(test_data_size))*100;
	test_loss/=(double)(test_data_size);

	print_test_result();
}


void network::send_signal_front(std::shared_ptr<std::vector<double>>input)
{
	layers[0]->connect_input(input);
	for (int i=0; i<layers_number;i++)
		layers[i]->get_output();
}

void network::send_signal_back(std::shared_ptr<std::vector<double>>expected_output)
{
	layers[layers_number-1]->get_error(expected_output);

	for (int i=layers_number-2;i>=0;i--)
		layers[i]->get_error();
}


void network::update_weights()
{ 	
	for (int i=0; i<layers_number;i++)
		layers[i]->update_weights();
}


void network::predict(std::shared_ptr<std::vector<double>>input)
{
	send_signal_front(input);
	matrix_and_vectors_operations::print_vector(layers[layers_number-1]->output);
}

void network::print_test_result()
{
	std::cout<<std::endl<<"test precision = "<<test_precision<<"% , test loss = "<<test_loss<<std::endl;
}

void network::print_validation_batch_result(int &batch_num)
{
	std::cout<<"samples: "<<batch_num*batch_size<<"/"<<batches_in_epoch*batch_size<<
		", validation loss: "<<validation_loss_batch<<std::endl;
}

void network::print_validation_epoch_result(int &epoch_num)
{
	std::cout<<"epoch: "<<epoch_num<<", mean validation loss: "<<validation_loss_epoch<<std::endl;
}

int network::get_batches_num_per_epoch(int &samples_num)
{
	int batches_in_epoch=0;
	batches_in_epoch=samples_num/batch_size;
	
	if(batches_in_epoch<1)
		batches_in_epoch=1;

	return batches_in_epoch;
}