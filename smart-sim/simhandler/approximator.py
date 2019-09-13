"""
Approximates datasets using an Artificial Neural Network

@author: Michael Bardwell, University of Alberta, Edmonton AB CAN
"""

from sklearn.neural_network import MLPRegressor
from extrema_detector import extrema_locator
import logging

# for testing -- delete soon
from n_dimensional_datasets import *
from plotter import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def adjusted_R2(R2, N, n):
    '''
    Adjusted R-squared as per formula https://en.wikipedia.org/wiki/Coefficient_of_determination#Adjusted_R2

    Parameters
    ----------
    R2: number-like

    N: int
        Number of samples
    n: int
        Number of features

    Returns
    -------
    float
    '''

    if not (isinstance(n, int) and isinstance(N, int)):
        raise TypeError("n-type: {}, N-type: {} must be int".format(type(n),
                                                                    type(N)))

    return 1-((1-R2)*(N-1)/(N-n-1))


def nd_reshape_for_MLPRegressor(x):
    '''
    Create input dataset from mesh

    Parameters
    ----------
    x: array-like
        Shape (n dimension, # samples dim n=0, ..., # samples dim n)

    Returns
    -------
    X: nested list
        Shape (n samples, n dimension). [[a a a], [b b b]] -> [[a b], [a b], [a b]]
    '''
    X = []
    X_sub = []
    for idx, _ in np.ndenumerate(x[0]):
        for dim in range(x.shape[0]):
            X_sub.append(x[dim][idx])
        X.append(X_sub)
        X_sub = []
    return X


class Approximator():
    '''
    Produces an approximation with a score of __% or above in __ seconds
    takes in [[a1 b1 ...], [a2 b2 ...]] and [z1, z2] an returns a trained
    three-layer MLPRegressor
    '''

    extrema_locator = extrema_locator

    def __init__(self, X, Y):
        if not (isinstance(X, np.ndarray) and isinstance(Y, np.ndarray)):
            raise TypeError("X and Y must be numpy arrays")

        if len(X.shape) != 2 or len(Y.shape) != 1:
            raise ValueError("Arrays have invalid shape")

        self.X = X
        self.Y = Y
        self.M = None

    def analyse_dataset(self):
        '''
        for extremely tight datasets (small y2-y1), eta will have to be
        smaller than for very loose datasets (large y2-y1). Problems will mostly
        arrise when eta is too large for the dataset, so we can take the stddev and
        divide it by 10000 to ensure there is enough headspace
        '''
        eta = Y.std()/10000
        self.M = len(extrema_locator(X, Y, eta))

    def setup_regression(self, ):
        self.regressor = MLPRegressor(
            hidden_layer_sizes=(n_hidden_layer+1,),
            activation='logistic',
            solver='lbfgs',
            alpha=0,
            max_iter=300,
            tol=1e-5,
            early_stopping=False,
            validation_fraction=0.1,
            n_iter_no_change=100,
            random_state=1
        )



if __name__ == "__main__":
    x = uniform_mesh(1, -1, 1, 0.05)
    f = decaying_sinewave_nd(x, 1.5)
    ext = mgrid_extremum_locator(x, f, 1e-3)
    M = len(ext)
    print("\nn: ", M+1)

    regressor = MLPRegressor(
        hidden_layer_sizes=(M+1,),
        activation='logistic',
        solver='lbfgs',
        alpha=0,
        max_iter=300,
        tol=1e-5,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=100,
        random_state=1
    )

    X = nd_reshape_for_MLPRegressor(x)
    F = f.reshape(-1)
    regressor.fit(X, F)

    logger.info("\nNumber of iterations: %s\nLoss^0.5: %f\nAdjusted R2: %f",
                regressor.n_iter_,
                np.sqrt(regressor.loss_),
                adjusted_R2(regressor.score(X, F), len(f), (M+1)))

    plotEnabled = True
    if plotEnabled and len(x) == 1:
        plot2d_extrema(x, f, ext)
        plot2d_approximation(x, f, regressor.predict(X))

    if plotEnabled and len(x) == 2:
        plot3d_extremums(x, f, ext)
        heatplot_extrema(x, f, ext)
