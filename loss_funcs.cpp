#include "stdafx.h"
#include "loss_funcs.h"

loss_funcs::loss_funcs(void)
{
}

loss_funcs::~loss_funcs(void)
{
}

std::shared_ptr<loss_funcs> loss_funcs::create_loss(loss_func_type lf_type)
{
	switch(lf_type)
	{
		case(loss_func_type::MSE):
			return std::make_shared<MSE_loss>();
		case(loss_func_type::CROSSENTROPY):
			return std::make_shared<CROSSENTROPY_loss>();
		default:
			return std::make_shared<MSE_loss>();
	}
}
