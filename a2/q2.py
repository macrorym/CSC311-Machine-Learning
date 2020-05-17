from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_data():
    boston = datasets.load_boston()
    X = boston.data
    y = boston.target
    features = boston.feature_names
    return X, y, features


def visualize(X, y, features):
    plt.figure(figsize=(20, 5))
    feature_count = X.shape[1]

    # i: index
    for i in range(feature_count):
        pt = plt.subplot(3, 5, i + 1)
        #TODO: Plot feature i against y
        pt.scatter(X[:,i], y)
        pt.set_xlabel(features[i])
        pt.set_ylabel('House Price in $1000')
        pt.set_title(features[i] + ' vs House Price')
    plt.tight_layout()
    plt.show()


def fit_regression(X, Y):
    #TODO: implement linear regression
    # Remember to use np.linalg.solve instead of inverting!
    weight = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(Y)
    return weight


def predict(x, weight):
    '''
    Return prediction of target with given weights
    :param x: numpy.ndarray
    :param weight: numpy.ndarray
    :return: list
    '''
    y = []
    for xi in x:
        y.append(np.dot(xi, weight))
    return y


def mse(y_hat, y):
    '''
    Return Mean Squared Error with given y hat and y
    :param y_hat: numpy.ndarray
    :param y: numpy.ndarray
    :return: float
    '''
    sum_error = (y - y_hat)**2
    return sum(sum_error)/len(y)


def mean_abs_error(y_hat, y):
    '''
    Return Mean Absolute Error with given y hat and y
    :param y_hat: numpy.ndarray
    :param y: numpy.ndarray
    :return: float
    '''
    error = abs(y - y_hat)
    return sum(error)/len(y)


def adjusted_r_sq(y_hat, y):
    '''
    Return Adjusted R square with given y hat and y
    :param y_hat: numpy.ndarray
    :param y: numpy.ndarray
    :return: float
    '''
    # R square
    r = 1 - (sum((y - y_hat)**2))/(sum((y - np.mean(y))**2))
    adjust = 1-(1-r)*(len(y)-1)/(len(y)-13-1)
    return adjust



def main():
    # Load the data
    X, y, features = load_data()
    print("Features: {}".format(features))
    
    # Visualize the features
    visualize(X, y, features)
    #TODO: Split data into train and test
    bias = []
    for i in range(506):
        bias.append([1])
    # Add bias term to data
    test = np.hstack((bias, X))
    x_train, x_test, y_train, y_test = train_test_split(test, y, train_size=0.7, random_state=9067)

    # Fit regression model
    w = fit_regression(x_train, y_train)
    # Compute fitted values, MSE, etc.
    y_hat = predict(x_test, w)
    print('Mean Squared Error:', mse(y_hat, y_test))
    # Mean absolute error measure average magnitude from predictions to obervations.
    print("Mean Absolute Error: ", mean_abs_error(y_hat, y_test))
    # Adjusted R Squared is great choice for measuring how much the data is explained by our Linear /
    # Regression Model. In this case, it represent the model performance too.
    print("Adjusted R-squared :", adjusted_r_sq(y_hat, y_test))

if __name__ == "__main__":
    main()

