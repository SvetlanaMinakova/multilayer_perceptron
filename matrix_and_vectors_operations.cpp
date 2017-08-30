#include "stdafx.h"
#include "matrix_and_vectors_operations.h"


matrix_and_vectors_operations::matrix_and_vectors_operations(void)
{
}


matrix_and_vectors_operations::~matrix_and_vectors_operations(void)
{
}

int matrix_and_vectors_operations::get_max_val_id(std::shared_ptr<std::vector<double>> vect)
{ int cur_id=0;
	  int max_val_id=0; 
	  double max_val=0;

	  for (double el : *vect)
	  {
		  if (el>max_val)
			{
				max_val=el;
				max_val_id=cur_id;
			}
		  cur_id++;
	  }

	return max_val_id;
}

void matrix_and_vectors_operations::set_zero_values(std::vector<double> &vect)
{
	for (int i=0;i<vect.size();i++)
		vect[i]=0;
}

//the best range of initialization by default is [0;1]
void matrix_and_vectors_operations::init_matrix_random(std::vector<std::vector<double>> &matrix)
{  
  	//fix the random generator to make the results reproduceable
	srand(7);
	double val;
	for(int j=0;j<matrix.size();j++)
	{
		for(int i=0;i<matrix[j].size();i++)
		{
			val= (double)(rand()%100)/100;
			matrix[j][i]=val;
		}
	}
}


//for some activation funcs exists special bounds for better weight matrix initialization
void  matrix_and_vectors_operations::init_matrix_random(std::vector<std::vector<double>> &matrix, double &scale_factor)
{   
	if(scale_factor==0)
		{//init by default
			init_matrix_random(matrix);
			return;
		}

	//fix the random generator to make the results reproduceable
	srand(7);
	double val;
	for(int j=0;j<matrix.size();j++)
	{
		for(int i=0;i<matrix[j].size();i++)
		{
			val= scale_factor * (double)(rand()%100)/100;
			if((int)(rand()%2)==1)
				{
					val*=(-1);
				}
			matrix[j][i]=val;
		}
	}
}


bool matrix_and_vectors_operations::classes_not_match(std::shared_ptr<std::vector<double>> vec1, std::shared_ptr<std::vector<double>> vec2)
{
	int max_val_id1= get_max_val_id(vec1);
	int max_val_id2=get_max_val_id(vec2);

	return max_val_id1!=max_val_id2;
}


void matrix_and_vectors_operations::print_vector(std::vector<double> &vect)
{	
	for_each(vect.begin(),vect.end(),[](double _x) 
	{std::cout<<std::setw(6)<<std::setprecision(4)<<std::fixed<<_x<<" ";});
	std::cout<<std::endl;
}

void matrix_and_vectors_operations::print_vector(std::shared_ptr<std::vector<double>> vect)
{	
	print_vector(*vect);
}

void matrix_and_vectors_operations::print_matrix(std::vector<std::vector<double>> &matrix)
{
	for (std::vector<double> row : matrix)
		{
			print_vector(row);
			std::cout<<std::endl;
		}
}

void matrix_and_vectors_operations::print_matrix(std::shared_ptr<std::vector<std::vector<double>>> matrix)
{
	print_matrix(*matrix);
}
