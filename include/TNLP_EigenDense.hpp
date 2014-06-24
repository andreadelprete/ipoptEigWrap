// Copyright (C) 2014, LAAS-CNRS
//
// Author: Andrea Del Prete     LAAS-CNRS    2014-06-24

#ifndef __TNLP_EIGEN_DENSE_HPP__
#define __TNLP_EIGEN_DENSE_HPP__

#include "IpTNLP.hpp"
#include <Eigen/Core>

namespace Eigen
{
    typedef Matrix<double,Dynamic,Dynamic,RowMajor> MatrixRXd;
};

/**
 *
 */
class TNLP_EigenDense : public Ipopt::TNLP
{
public:
    /** default constructor */
    TNLP_EigenDense();
    
    /** default destructor */
    virtual ~TNLP_EigenDense();
    
    /**@name Overloaded from TNLP */
    //@{
    /** Method to return some info about the nlp */
    bool get_nlp_info(Ipopt::Index& n, Ipopt::Index& m, Ipopt::Index& nnz_jac_g,
                              Ipopt::Index& nnz_h_lag, IndexStyleEnum& index_style);
    virtual bool get_nlp_info(int n, int m) = 0;
    
    
    /** Method to return the bounds for my problem */
    bool get_bounds_info(Ipopt::Index n, Ipopt::Number* x_l, Ipopt::Number* x_u,
                                 Ipopt::Index m, Ipopt::Number* g_l, Ipopt::Number* g_u);
    /** Method to return the bounds for my problem */
    virtual bool get_bounds_info(int n, Eigen::VectorXd &x_l, Eigen::VectorXd &x_u,
                                 int m, Eigen::VectorXd &g_l, Eigen::VectorXd &g_u) = 0;
    
    /** Method to return the starting point for the algorithm */
    bool get_starting_point(Ipopt::Index n, bool init_x, Ipopt::Number* x,
                            bool init_z, Ipopt::Number* z_L, Ipopt::Number* z_U,
                            Ipopt::Index m, bool init_lambda, Ipopt::Number* lambda);
    /** Method to return the starting point for the algorithm */
    virtual bool get_starting_point(int n, bool init_x, Eigen::VectorXd &x,
                                    bool init_z, Eigen::VectorXd &z_L, Eigen::VectorXd &z_U,
                                    int m, bool init_lambda, Eigen::VectorXd &lambda) = 0;
    
    /** Method to return the objective value */
    bool eval_f(Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Number& obj_value);
    /** Method to return the objective value */
    virtual bool eval_f(int n, const Eigen::VectorXd &x, bool new_x, double &obj_value) = 0;
    
    /** Method to return the gradient of the objective */
    bool eval_grad_f(Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Number* grad_f);
    /** Method to return the gradient of the objective */
    virtual bool eval_grad_f(int n, const Eigen::VectorXd &x, bool new_x, Eigen::VectorXd &grad_f) = 0;
    
    /** Method to return the constraint residuals */
    bool eval_g(Ipopt::Index n, const Ipopt::Number* x, bool new_x, Ipopt::Index m, Ipopt::Number* g);
    /** Method to return the constraint residuals */
    virtual bool eval_g(int n, const Eigen::VectorXd &x, bool new_x, int m, Eigen::VectorXd &g) = 0;
    
    /** Method to return:
     *   1) The structure of the jacobian (if "values" is NULL)
     *   2) The values of the jacobian (if "values" is not NULL)
     */
    bool eval_jac_g(Ipopt::Index n, const Ipopt::Number* x, bool new_x,
                    Ipopt::Index m, Ipopt::Index nele_jac, Ipopt::Index* iRow, Ipopt::Index *jCol,
                    Ipopt::Number* values);
    /** Method to return the values of the Jacobian. */
    virtual bool eval_jac_g(int n, const Eigen::VectorXd &x, bool new_x,
                            int m, Eigen::MatrixRXd &values) = 0;
    
    /** Method to return:
     *   1) The structure of the hessian of the lagrangian (if "values" is NULL)
     *   2) The values of the hessian of the lagrangian (if "values" is not NULL)
     */
    bool eval_h(Ipopt::Index n, const Ipopt::Number* x, bool new_x,
                        Ipopt::Number obj_factor, Ipopt::Index m, const Ipopt::Number* lambda,
                        bool new_lambda, Ipopt::Index nele_hess, Ipopt::Index* iRow,
                        Ipopt::Index* jCol, Ipopt::Number* values);
    /** Method to return the values of the hessian of the lagrangian. */
    virtual bool eval_h(int n, const Eigen::VectorXd &x, bool new_x,
                        double obj_factor, int m, const Eigen::VectorXd &lambda,
                        bool new_lambda, Eigen::MatrixRXd &values) = 0;
    
    //@}
    
    /** @name Solution Methods */
    //@{
    /** This method is called when the algorithm is complete so the TNLP can store/write the solution */
    void finalize_solution(Ipopt::SolverReturn status,
                                   Ipopt::Index n, const Ipopt::Number* x, const Ipopt::Number* z_L, const Ipopt::Number* z_U,
                                   Ipopt::Index m, const Ipopt::Number* g, const Ipopt::Number* lambda,
                                   Ipopt::Number obj_value,
                                   const Ipopt::IpoptData* ip_data,
                                   Ipopt::IpoptCalculatedQuantities* ip_cq);
    //@}
    /** This method is called when the algorithm is complete so the TNLP can store/write the solution */
    virtual void finalize_solution(Ipopt::SolverReturn status,
                                   int n, const Eigen::VectorXd &x,
                                   const Eigen::VectorXd &z_L, const Eigen::VectorXd &z_U,
                                   int m, const Eigen::VectorXd &g, const Eigen::VectorXd &lambda,
                                   double obj_value,
                                   const Ipopt::IpoptData* ip_data,
                                   Ipopt::IpoptCalculatedQuantities* ip_cq) = 0;
    
private:
    /**@name Methods to block default compiler methods.
     * The compiler automatically generates the following three methods.
     *  Since the default compiler implementation is generally not what
     *  you want (for all but the most simple classes), we usually
     *  put the declarations of these methods in the private section
     *  and never implement them. This prevents the compiler from
     *  implementing an incorrect "default" behavior without us
     *  knowing. (See Scott Meyers book, "Effective C++")
     *  
     */
    //@{
    //  TNLP_EigenDense();
    TNLP_EigenDense(const TNLP_EigenDense&);
    TNLP_EigenDense& operator=(const TNLP_EigenDense&);
    //@}
};


#endif
