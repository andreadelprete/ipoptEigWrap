// Copyright (C) 2014, LAAS-CNRS
//
// Author: Andrea Del Prete     LAAS-CNRS    2014-06-24

#include "TNLP_EigenDense.hpp"

#include <cassert>
#include <iostream>

using namespace Ipopt;
using namespace Eigen;

// constructor
TNLP_EigenDense::TNLP_EigenDense()
{}

//destructor
TNLP_EigenDense::~TNLP_EigenDense()
{}

// returns the size of the problem
bool TNLP_EigenDense::get_nlp_info(Index& n, Index& m, Index& nnz_jac_g,
                                   Index& nnz_h_lag, IndexStyleEnum& index_style)
{
    bool res = this->get_nlp_info(n,m);
    // assume Jacobian is dense
    nnz_jac_g = n*m;
    // Hessian is also dense
    nnz_h_lag = n*n;
    // use the C style indexing (0-based)
    index_style = TNLP::C_STYLE;
    return res;
}

// returns the variable bounds
bool TNLP_EigenDense::get_bounds_info(Index n, Number* x_l, Number* x_u,
                                      Index m, Number* g_l, Number* g_u)
{
    Map<VectorXd> x_l_eig(x_l,n);
    Map<VectorXd> x_u_eig(x_u,n);
    Map<VectorXd> g_l_eig(g_l,m);
    Map<VectorXd> g_u_eig(g_u,m);
    return get_bounds_info(x_l_eig, x_u_eig, g_l_eig, g_u_eig);
    
    // Ipopt interprets any number greater than nlp_upper_bound_inf as
    // infinity. The default value of nlp_upper_bound_inf and nlp_lower_bound_inf
    // is 1e19 and can be changed through ipopt options.
}

// returns the initial point for the problem
bool TNLP_EigenDense::get_starting_point(Index n, bool init_x, Number* x,
                                         bool init_z, Number* z_L, Number* z_U,
                                         Index m, bool init_lambda, Number* lambda)
{
    Map<VectorXd> x_eig(x,n);
    Map<VectorXd> z_L_eig(z_L,n);
    Map<VectorXd> z_U_eig(z_U,n);
    Map<VectorXd> lambda_eig(lambda,m);
    return get_starting_point(init_x, x_eig, init_z, z_L_eig, z_U_eig,
                              init_lambda, lambda_eig);
}

// returns the value of the objective function
bool TNLP_EigenDense::eval_f(Index n, const Number* x, bool new_x, Number& obj_value)
{
    const Map<const VectorXd> x_eig(x,n);
    return eval_f(x_eig, new_x, obj_value);
}

// return the gradient of the objective function grad_{x} f(x)
bool TNLP_EigenDense::eval_grad_f(Index n, const Number* x, bool new_x, Number* grad_f)
{
    const Map<const VectorXd> x_eig(x,n);
    Map<VectorXd> grad_f_eig(grad_f,n);
    return eval_grad_f(x_eig, new_x, grad_f_eig);
}

// return the value of the constraints: g(x)
bool TNLP_EigenDense::eval_g(Index n, const Number* x, bool new_x, Index m, Number* g)
{
    const Map<const VectorXd> x_eig(x,n);
    Map<VectorXd> g_eig(g,m);
    return eval_g(x_eig, new_x, g_eig);
}

// return the structure or values of the jacobian
bool TNLP_EigenDense::eval_jac_g(Index n, const Number* x, bool new_x,
                                 Index m, Index nele_jac, Index* iRow, Index *jCol,
                                 Number* values)
{
    if (values == NULL)
    {
        // return the structure of the jacobian assuming it's dense
        int el = 0;
        for(int i=0; i<m; i++)
        {
            for(int j=0; j<n; j++)
            {
                iRow[el] = i;
                jCol[el] = j;
                el++;
            }
        }
        return true;
    }
    
    // return the values of the jacobian of the constraints
    const Map<const VectorXd> x_eig(x,n);
    Map<MatrixRXd> values_eig(values,m,n);
    return eval_jac_g(x_eig, new_x, values_eig);
}

//return the structure or values of the hessian
bool TNLP_EigenDense::eval_h(Index n, const Number* x, bool new_x,
                             Number obj_factor, Index m, const Number* lambda,
                             bool new_lambda, Index nele_hess, Index* iRow,
                             Index* jCol, Number* values)
{
    if (values == NULL)
    {
        // return the structure. This is a symmetric matrix, fill the lower left
        // triangle only. Assume the hessian is dense
        Index idx=0;
        for (Index row = 0; row < n; row++)
        {
            for (Index col = 0; col < n; col++)
            {
                iRow[idx] = row;
                jCol[idx] = col;
                idx++;
            }
        }
        assert(idx == nele_hess);
        return true;
    }
    
    const Map<const VectorXd> x_eig(x,n);
    const Map<const VectorXd> lambda_eig(lambda,m);

    // @todo "values" should be mapped into a lower triangular matrix
    Map<MatrixRXd> values_eig(values,n,n);
    
    return eval_h(x_eig, new_x, obj_factor, lambda_eig, new_lambda, values_eig);
}

void TNLP_EigenDense::finalize_solution(SolverReturn status,
                                        Index n, const Number* x, const Number* z_L, const Number* z_U,
                                        Index m, const Number* g, const Number* lambda,
                                        Number obj_value,
                                        const IpoptData* ip_data,
                                        IpoptCalculatedQuantities* ip_cq)
{
    const Map<const VectorXd> x_eig(     x,      n);
    const Map<const VectorXd> z_L_eig(   z_L,    n);
    const Map<const VectorXd> z_U_eig(   z_U,    n);
    const Map<const VectorXd> g_eig(     g,      m);
    const Map<const VectorXd> lambda_eig(lambda, m);
    finalize_solution(status, x_eig, z_L_eig, z_U_eig, g_eig, lambda_eig, obj_value, ip_data, ip_cq);
}
