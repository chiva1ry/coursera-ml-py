import numpy as np


def compute_cost(X, y, theta):
    # Initialize some useful values
    m = y.size
    cost = 0

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta.
    #                You should set the variable "cost" to the correct value.
    
    # np.dot matrix multiplication
    cost = np.sum((np.dot(X, theta)-y)**2)/(2*m)

    # ==========================================================

    return cost
