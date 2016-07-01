# -*- coding: utf-8 -*-
"""
Spyder Editor
University Washington Machine Learning Course 2
using SFrame 
This is a temporary script file.
"""

import sframe
import numpy as np

dtype_dict = {'bathrooms': float,
              'waterfront':int,
              'sqft_above': int,
              'sqft_living15': float,
              'grade': int,
              'yr_renovated': int,
              'price':float,
              'bedrooms': float,
              'zipcode': str,
              'long': float,
              'sqft_lot15': float,
              'sqft_living': float,
              'floors': str,
              'condition': int,
              'lat': float,
              'date': str,
              'sqft_basement': int,
              'yr_built': int,
              'id': str,
              'sqft_lot': int,
              'view': int}
sales = sframe.SFrame.read_csv('kc_house_data.csv',
                               column_type_hints=dtype_dict )
sales['bedrooms_squared'] = sales['bedrooms'] * sales['bedrooms']
sales['bed_bath_rooms'] = sales['bedrooms'] * sales['bathrooms']
sales['log_sqft_living'] = np.log(sales['sqft_living'])
sales['lat_long_plus'] = sales['lat'] + sales['long']
train_data, test_data = sales.random_split(.8,seed=0)


def get_numpy_data(data_sframe, features, output):
    data_sframe['constant'] = 1 # add a constant column to an SFrame
    # prepend variable 'constant' to the features list
    features = ['constant'] + features
    # select the columns of data_SFrame given by the ‘features’ list into the SFrame ‘features_sframe’

    # this will convert the features_sframe into a numpy matrix
    features_matrix = data_sframe[features].to_numpy()
    # assign the column of data_sframe associated with the target to the variable ‘output_sarray’

    # this will convert the SArray into a numpy array:
    output_array = data_sframe[output].to_numpy() # GraphLab Create>= 1.7!!
    return(features_matrix, output_array) 


def predict_outcome(feature_matrix, weights):
    predictions = np.dot(feature_matrix, weights)
    return(predictions)
    

def feature_derivative(errors, feature):
    derivative = 2 * np.dot(feature, errors)
    return(derivative)


def regression_gradient_descent(feature_matrix, output, initial_weights,
                                step_size, tolerance):
    converged = False
    weights = np.array(initial_weights)
    while not converged:
        # compute the predictions based on feature_matrix and weights:
        predictions = predict_outcome(feature_matrix, weights)
        # compute the errors as predictions - output:
        errors = predictions - output        
        gradient_sum_squares = 0 # initialize the gradient
        # while not converged, update each weight individually:
        for i in range(len(weights)):
            # Recall that feature_matrix[:, i] is the feature column associated with weights[i]
            # compute the derivative for weight[i]:
            derivative = feature_derivative(errors, feature_matrix[:,i])
            # add the squared derivative to the gradient magnitude
            gradient_sum_squares += derivative**2
            # update the weight based on step size and derivative:
            weights[i] -= step_size * derivative
        gradient_magnitude = np.sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance:
            converged = True
    return(weights)


def get_residual_sum_of_squares(model, data, outcome):
    # First get the predictions
    predictions = predict_outcome(data, model)
    # Then compute the residuals/errors
    errors = predictions - outcome
    # Then square and add them up
    RSS = np.sum(errors**2)
    return(RSS)    
    
if __name__ == '__main__':
    print train_data
    simple_features = ['sqft_living']
    my_output= 'price'
    (simple_features_matrix, output) = get_numpy_data(train_data, simple_features,
                                                   my_output)
    initial_weights = np.array([-47000., 1.])
    step_size = 7e-12
    tolerance = 2.5e7
    simple_weights = regression_gradient_descent(
        simple_features_matrix, output, initial_weights, step_size, tolerance)
    print simple_weights

    # example_features = ['sqft_living', 'bedrooms', 'bathrooms']
