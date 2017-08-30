#pragma once
#include <vector>
#include <iostream>
#include <ctime>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <memory>
class matrix_and_vectors_operations
{
public:
	matrix_and_vectors_operations(void);
	~matrix_and_vectors_operations(void);
	static void print_vector(std::vector<double> &vect);
	static void print_vector(std::shared_ptr<std::vector<double>> vect);
	static void print_matrix(std::vector<std::vector<double>> &matrix);
	static void print_matrix(std::shared_ptr<std::vector<std::vector<double>>> matrix);

	static void init_matrix_random(std::vector<std::vector<double>> &matrix);
	static void init_matrix_random(std::vector<std::vector<double>> &matrix, double &scale_factor);

	static int get_max_val_id (std::shared_ptr<std::vector<double>> vect);
	static void set_zero_values(std::vector<double> &vect);
	static bool classes_not_match(std::shared_ptr<std::vector<double>> vec1, std::shared_ptr<std::vector<double>> vec2);
	static std::shared_ptr<std::vector<double>> transform_to_binary_vector(int class_id);
	static std::shared_ptr<std::vector<std::shared_ptr<std::vector<double>>>> 
	generate_binary_vecors_range (int classes_number);
};


