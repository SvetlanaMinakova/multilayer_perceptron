#include "stdafx.h"
#include "input_data_manager.h"
#include <algorithm>
#define DATA_FOLDER_PATH "InputTxtData/"

void input_data_manager::create_samples(std::vector<std::shared_ptr<std::vector<double>>>& source_data_vectors,
	std::vector<std::shared_ptr<std::vector<double>>>& source_labels,
	std::vector<std::shared_ptr<std::vector<double>>>& result_x_placeholder,
	std::vector<std::shared_ptr<std::vector<double>>>& result_y_placeholder,
	int x_len,int samples_num,double noise_level)
{

	result_x_placeholder.reserve(samples_num);
	result_y_placeholder.reserve(samples_num);

	//seed for random generator to make the result repeateable
	srand(42);

	int possible_inputs_num=source_data_vectors.size();
	int pixels_for_changing_num =(int)(noise_level*double(samples_num));

	//data shuffling and augmentation
	int source_inp_id=0;
	std::shared_ptr<std::vector<double>> cur_source_inp;
	for(int s=0; s<samples_num;++s)
	{
		source_inp_id=(int)(rand()%possible_inputs_num);
		std::shared_ptr<std::vector<double>> cur_x=std::make_shared<std::vector<double>>();
		cur_x->resize(x_len);
		//copy the "perfect" input data
		cur_source_inp=source_data_vectors[source_inp_id];
		for(int i=0; i<x_len;++i)
			cur_x->at(i)=cur_source_inp->at(i);
	
		//"spoil" the sample with noise
		int changing_pixel_id;
		int noised_pixel_value;
		
		for(int i=0; i<pixels_for_changing_num;++i)
			{
				changing_pixel_id=(rand()%x_len);
				noised_pixel_value = (double)(rand()%100)/1000;
				cur_x->at(changing_pixel_id)=noised_pixel_value;
			}

		//push noised_x and (unchanged) y into result vectors
		result_x_placeholder.push_back(cur_x);
		result_y_placeholder.push_back(source_labels[source_inp_id]);
	}
}

std::vector<std::shared_ptr<std::vector<double>>> input_data_manager::get_all_data_from_InputTxtData_folder()
{
	setlocale(LC_ALL, "rus");
    WIN32_FIND_DATA findf;
	LPCWSTR str = _T("InputTxtData/*.txt");
	std::string filename="";
	char ch=' ';
	int id=0;

	std::vector<std::shared_ptr<std::vector<double>>> all_data;

    HANDLE hFind = FindFirstFile(str, &findf);
		while(ch!='\0')
	   {
			ch=(char)(findf.cFileName[id]);
			filename+= ch;
			id++;
		}

		all_data.push_back(get_vector_from_file(DATA_FOLDER_PATH + filename));

    while (FindNextFile(hFind, &findf)){
		filename="";
		ch=' ';
		id=0;
	while(ch!='\0')
		{
			ch=(char)(findf.cFileName[id]);
			filename+= ch;
			id++;
		}
		all_data.push_back(get_vector_from_file(DATA_FOLDER_PATH + filename));
    }
//end search
    FindClose(hFind);	

	return all_data;
}

std::shared_ptr<std::vector<double>> input_data_manager::get_vector_from_file(std::string filename,bool normalization)
{
	std::ifstream fin;
	std::string s="";
	std::string temp;

	fin.open(filename,std::ios_base::in);
	while(fin>>temp)
		s+=temp;
    fin.close();

	std::shared_ptr<std::vector<double>> res = std::make_shared<std::vector<double>>();
	int strid=0;
	int strlen=s.size();
	temp="";
	
	while (strid<strlen)
	{

		while(s[strid]!=',')
		{
			temp+=s[strid];
		  strid++;
		}
		//get img pixels,[0...255] grey colors
		res->push_back(atof(temp.c_str())/254);

		temp="";
	strid++;
	}

	if(normalization)
	{
	//input normalization
	for(int i=0; i< res->size();i++)
		res->at(i) = res->at(i)*2-1;
	}
return res;
}

std::vector<std::shared_ptr<std::vector<double>>> input_data_manager::get_binary_labels(int classes_num)
{
	std::vector<std::shared_ptr<std::vector<double>>> result;
	result.reserve(classes_num);

	for(int i=0;i<classes_num;i++)
		result.push_back(get_binary_class_vector(i,classes_num));

	return result;
}

std::shared_ptr<std::vector<double>> input_data_manager::get_binary_class_vector(int class_id,int classes_num)
{
	std::shared_ptr<std::vector<double>> result=std::make_shared<std::vector<double>>();
	result->resize(classes_num);
	result->at(class_id)=1;
	//numeration from the smallest values
	std::reverse(result->begin(),result->end());
	return result;
}

int input_data_manager::get_input_data_len(std::vector<std::shared_ptr<std::vector<double>>> &source_data_vectors,bool check_that_len_is_min)
{
	int data_len=0;
	
	if(source_data_vectors.size()>0)
		data_len=source_data_vectors[0]->size();

	if(check_that_len_is_min)
	{
		for(auto data_vect : source_data_vectors)
		{
			if(data_len>data_vect->size())
				data_len=data_vect->size();
		}
	}
	return data_len;
}


input_data_manager::input_data_manager(void)
{}

input_data_manager::~input_data_manager(void)
{}
