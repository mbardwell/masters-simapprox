import numpy as np
import operator


# @brief r2_adj: adjusted R-squared as per formula https://en.wikipedia.org/wiki/Coefficient_of_determination#Adjusted_R2
# @param R2: float
# @param N: int. Number of samples
# @param n: int. Number of features
# @returns: float
def r2_adj(R2, N, n):
    return 1-((1-R2)*(N-1)/(N-n-1))

# @brief rmse: root-mean squared error as per formula https://en.wikipedia.org/wiki/Root-mean-square_deviation
# @param prediction: numpy array. Predicted values of function
# @param target: numpy array. Actual values of function
# @returns: float
def rmse(prediction, target):
    return np.sqrt(np.mean(np.square(list(map(operator.sub, prediction, target)))))