#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================================================================================================
# Created By  : Thiago Mergulhão - University of Edinburgh
# Created Date: 2022-02-15 10:28:29
# ==================================================================================================
"""This code marginalise over linear parameters of the likelihood
# It works with any model.
"""
# ==================================================================================================
# Imports
# ==================================================================================================

import numpy as np

# ==================================================================================================
# Functions to get the part of the model that is constant and linear on the parameters.
# ==================================================================================================

def get_const_part(model, params, linear_indices, args = None):
    """Get the constant part of the model (i.e the part of the theory that does not depend on the 
    linear parameters).

    Args:
        model (function): The theoretical model being used. It must be implemented such that its in-
        puts are of the form: model(params, args(optional))
        
        params (list): A list with the parameters values
        
        linear_indices (list): A list with the indexes of the position of the linear parameters in 
        the params list.
        
        args (tuple, optional): Additional arguments used to compute the model. 
        Defaults to None.

    Returns:
        np.array: The constant part of the theory 
    """
    temp_params = params.copy()
    
    for lin_index in linear_indices:
        temp_params[lin_index] = 0
        
    if args is not None:
        return model(temp_params, args = args)
    
    else:
        return model(temp_params)
        
def get_lin_operator_list(model, params, linear_indices, args = None):
    """Get all the linear contributions to the theory

    Args:
        model (function): The theoretical model being used. It must be implemented such that its in-
        puts are of the form: model(params, args(optional))
        
        params (list): A list with the parameters values
        
        linear_indices (list): A list with the indexes of the position of the linear parameters in 
        the params list.
        
        args (tuple, optional): Additional arguments used to compute the model. 
        Defaults to None.

    Returns:
        list: A list with the linear contributions computed for the given set of parameters
    """
    
    #Compute the full theoretical model
    lin_operator_list = []
    
    for this_lin_index in linear_indices:
        params_partial = params.copy()
        params_partial[this_lin_index] = 0
        
        if args is not None:
            model_full    = model(params, args)
            model_partial = model(params_partial, args)
        else:
            model_full    = model(params)
            model_partial = model(params_partial)
        output = (model_full - model_partial)/params[this_lin_index]
        lin_operator_list.append(np.asarray(output))
    return lin_operator_list

# ==================================================================================================
# Computing the marginalisation matrices
# ==================================================================================================

def QuadraticForm(A,B,M):
    """Compute a quadratic form: A.M.B^{T}

    Args:
        A (np.array): The first vector, should be n x 1
        B (np.array): The second vector, should be n x 1
        invCOV (np.array): The matrix. Should be n X n

    Returns:
        float: The result of the quadratic form
    """

    return np.dot(A, np.dot(M,B.T))

def compute_Aij(theory_lin_list, inv_cov):
    """ Compute the matrix A_ij. See notes for its definition

    Args:
        theory_lin_list(list): A list with all contributions of the linear parameters to the theory

        inv_cov(np.array): The inverse of the covariance matrix

    Returns:
        np.array: The matrix A_ij. It is a (Np x Np) matrix, where Np is the number of linear para-
        meters that were marginalised.
    """
    
    #Get the number of linear parameters to be marginalised
    lin_dim = len(theory_lin_list)
    A_ij = np.zeros((lin_dim, lin_dim))
    
    #Construct the matrix
    for i in range(0, lin_dim):
        for j in range(0, lin_dim):
            A_ij[i, j] = QuadraticForm(theory_lin_list[i], theory_lin_list[j], inv_cov)
    return A_ij

def compute_Bi(theory_lin_list, theory_const, data, inv_cov):
    """ Compute the matrix B_i. See notes for its definition

    Args:
        theory_lin_list(list of arrays): A list with all contributions of the linear parameters to 
        the theory

        theory_const (np.array): The part of the theory that does not depend on the linear parame-
        ters.
           
        data (np.array: The data vector.
        
        inv_cov (np.array): The inverse of the covariance matrix.

    Returns:
        np.array: The matrix B_i. It is a (1 x Np) matrix, where Np is the number of linear para-
        meters that were marginalised.
    """

    lin_dim = len(theory_lin_list)

    #Initialize the output
    B_i = np.zeros((1, lin_dim))
    
    #Compute each component of B_i separately 
    for i in range(0, lin_dim):
        term1 = QuadraticForm(theory_const, theory_lin_list[i], inv_cov)
        term2 = QuadraticForm(data, theory_lin_list[i], inv_cov)
        term3 = QuadraticForm(theory_lin_list[i], data, inv_cov)
        term4 = QuadraticForm(theory_lin_list[i], theory_const, inv_cov)
        B_i[0,i] = -0.5*(term1 - term2 - term3 + term4)
    return B_i

def compute_C(theory_const, data, inv_cov):
    """ Compute the term C. See notes for its definition

    Args:
        theory_const (np.array): The part of the theory that does not depend on the linear parame-
        ters.

        data (np.array: The data vector.
        
        inv_cov (np.array): The inverse of the covariance matrix.

    Returns:
        np.array: The term C
    """
    term1 = QuadraticForm(data, data, inv_cov)
    term2 = QuadraticForm(data, theory_const, inv_cov)
    term3 = QuadraticForm(theory_const, data, inv_cov)
    term4 = QuadraticForm(theory_const, theory_const, inv_cov)  
    return -0.5*(term1 - term2 - term3 + term4)

# ==================================================================================================
# Computing the marginalised likelihood
# ==================================================================================================

def loglikelihood_marginalised(theta, theta_marginalised, flat_prior, linear_indices, 
                               model, data, inv_cov, theory_args = None):
    """Compute the marginalised likelihood

    Args:
        theta (list): A list with the remaining free parameters in the model. These paramters change
        on every step of the MCMC analysis.
        
        theta_marginalised (list): Some exemplary values for the linear parameters you marginalised
        . They are used to get the constant and linear part on every step. These parameters are fi-
        xed, they do not vary on every step of the MCMC analysis.
        
        flat_prior (function): The flat prior for the model.
        
        linear_indices (list): A list with the indices where the linear parameters you are margina-
        lising occur in the total param array
       
        model (function): The theoretical model being used. It must be implemented such that its in-
        puts are of the form: model(params, args(optional)).
        
        data (np.array): The data vector.
        
        inv_cov (np.array): The inverse of the covariance matrix
        
        theory_args (_type_, optional): _description_. Defaults to None.

    Returns:
        float: The output for the marginalised log likelihood
    """

    #Apply a flat prior to some parameters
    if not(flat_prior(theta)):
        return(-np.inf)
    
    params = np.hstack((theta, theta_marginalised))
    
    #Get linear terms
    linear_operators_list = get_lin_operator_list(model, params, linear_indices, args = theory_args)
    
    #Get the constant part
    theory_const = get_const_part(model, params, linear_indices, args = theory_args)
    
    #Compute the matrices
    A_ij = compute_Aij(linear_operators_list, inv_cov)
    
    B_i  = compute_Bi(linear_operators_list, theory_const, data, inv_cov)
    
    C    = compute_C(theory_const, data, inv_cov)
    
    #Get the inverse of A
    invAij = np.linalg.inv(A_ij)
    
    #Get the determinant of A
    detAij = np.linalg.det(A_ij)
    
    #Compute the marginalised likelihood
    term1 = QuadraticForm(B_i, B_i, invAij); term2 = np.log(detAij)
    return (0.5*term1 - 0.5*term2 + C)[0][0]