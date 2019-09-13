import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer

class DataFrameSelector(BaseEstimator, TransformerMixin):
    """
    From Hand on Machine Learning - A. Geron
    """
    def __init__(self, attribute_names):
        self.attribute_names=attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
    
class MyLabelBinarizer(BaseEstimator, TransformerMixin):
    """
    https://stackoverflow.com/questions/46162855/fit-transform-takes-2-positional-arguments-but-3-were-given-with-labelbinarize#46165319
    """
    def __init__(self, sparse_output=False):
        self.sparse_output = sparse_output
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        enc = LabelBinarizer(sparse_output=self.sparse_output)
        return enc.fit_transform(X)
    
class RejectOutliers(BaseEstimator, TransformerMixin):
    """
    Finds outliers in given dimensions and remove data point from all dimensions in data
    
    parameters
    ----------
    outlier_attributes: list. Dimensions to evaluate for outliers
        ex: ["vmag-671", "vmag-632"]
    m: int. Number of std deviations from mean before point is considered outlier
    """
    def reject_outliers(self, data, m=3, return_positions=False):
        positions = abs(data - np.mean(data)) < m * np.std(data)
        if return_positions:
            return positions
        return data[positions]
    def __init__(self, outlier_attributes, m=3):
        self.outlier_attributes = outlier_attributes
        self.m = m
    def fit(self, data):
        return self
    def transform(self, data):
        for y in self.outlier_attributes:
            outlier_positions = self.reject_outliers(data[y], self.m, return_positions=True)
            data = data[outlier_positions]
        return data