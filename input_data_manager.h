#pragma once
#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <io.h>
#include <string>
#include <vector>
#include <windows.h> 
#include <memory>

class input_data_manager
{
public:
	input_data_manager(void);
	~input_data_manager(void);
	static std::vector<std::shared_ptr<std::vector<double>>> get_all_data_from_InputTxtData_folder();
	static std::vector<std::shared_ptr<std::vector<double>>> get_binary_labels(int classes_num);
	static int get_input_data_len(std::vector<std::shared_ptr<std::vector<double>>> &source_data_vectors, bool check_that_len_is_min=false);

	static void create_samples(std::vector<std::shared_ptr<std::vector<double>>>& source_data_vectors,
	std::vector<std::shared_ptr<std::vector<double>>> &source_labels,
	std::vector<std::shared_ptr<std::vector<double>>> &result_x_placeholder,
	std::vector<std::shared_ptr<std::vector<double>>> &result_y_placeholder,
	int x_len,int samples_num,double noise_level=0);

private:
	static std::shared_ptr<std::vector<double>> get_vector_from_file(std::string filename,bool normalization=false);
	static std::shared_ptr<std::vector<double>> get_binary_class_vector(int class_id,int classes_num);
};

