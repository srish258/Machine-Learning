import numpy as np
import pandas as pd

############################################################################
# DO NOT MODIFY CODES ABOVE 
# DO NOT CHANGE THE INPUT AND OUTPUT FORMAT
############################################################################

###### Part 1.1 ######
def mean_square_error(w, X, y):
    """
    Compute the mean square error of a model parameter w on a test set X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test features
    - y: A numpy array of shape (num_samples, ) containing test labels
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean square error
    """
    #####################################################
    # TODO 1: Fill in your code here                    #
    #####################################################
    err = np.square(np.abs(np.subtract(np.dot(X, w), np.array(y, np.float64)))).mean() 
    
    return err

###### Part 1.2 ######
def linear_regression_noreg(X, y):
  """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing features
  - y: A numpy array of shape (num_samples, ) containing labels
  Returns:
  - w: a numpy array of shape (D, )
  """
  #####################################################
  #	TODO 2: Fill in your code here                    #
  w = None
  x_transpose_x = np.matmul(np.transpose(X), X)
  inv_xtx = np.linalg.inv(x_transpose_x)
  x_transpose_y = np.matmul(np.transpose(X), y)
  w = np.matmul(inv_xtx, x_transpose_y)
  #####################################################		
  
  return w


###### Part 1.3 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing features
    - y: A numpy array of shape (num_samples, ) containing labels
    - lambd: a float number specifying the regularization parameter
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here                    #
    x_transpose_x = np.dot(np.transpose(X), X)
    regularized_x_transpose_x = np.add(x_transpose_x, np.identity(len(x_transpose_x)) * lambd)
    inv_x = np.linalg.inv(regularized_x_transpose_x)
    x_transpose_y = np.dot(np.transpose(X), y)
    w = np.dot(inv_x, x_transpose_y)
  #####################################################		
    
    return w

###### Part 1.4 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training features
    - ytrain: A numpy array of shape (num_training_samples, ) containing training labels
    - Xval: A numpy array of shape (num_val_samples, D) containing validation features
    - yval: A numpy array of shape (num_val_samples, ) containing validation labels
    Returns:
    - bestlambda: the best lambda you find among 2^{-14}, 2^{-13}, ..., 2^{-1}, 1.
    """
    #####################################################
    # TODO 5: Fill in your code here                    #
    lambds = [2 ** -14, 2 ** -13, 2 ** -12, 2 ** -11,2 ** -10, 2 ** -9, 2 ** -8, 2 ** -7, 2 ** -6, 2 ** -5, 2 ** -4, 2 ** -3, 2 ** -2, 2 ** -1, 1]
    elist = np.zeros((1, 1))
    for i in range(0, len(lambds)):
        wtemp = regularized_linear_regression(Xtrain, ytrain, lambds[i])
        ycal = Xval @ wtemp
        error = np.power((yval - ycal), 2)
        avgerror = np.average(error)
        elist = np.append(elist, avgerror)
    elist = np.delete(elist, 0, 0)
    bestlambdaindex = np.argmin(elist)
    bestlambda = lambds[bestlambdaindex]
    #####################################################		
    
    return bestlambda
    

###### Part 1.6 ######
def mapping_data(X, p):
    """
    Augment the data to [X, X^2, ..., X^p]
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training features
    - p: An integer that indicates the degree of the polynomial regression
    Returns:
    - X: The augmented dataset. You might find np.insert useful.
    """
    #####################################################
    # TODO 6: Fill in your code here                    #
    i = 1
    multiplied_X, mapped_X_matrix = X, X
    while i < p:
        multiplied_X = np.multiply(multiplied_X, X)
        mapped_X_matrix = np.append(mapped_X_matrix, multiplied_X, axis=1)
        i += 1
    #####################################################		
    
    return mapped_X_matrix

"""
NO MODIFICATIONS below this line.
You should only write your code in the above functions.
"""

