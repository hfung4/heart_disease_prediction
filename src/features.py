# src/features.py
# Contains custom pipeline steps
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
import numpy as np
from typing import List
from pathlib import Path
import pickle


class Mapper(BaseEstimator, TransformerMixin):
    """
    Constructor

    Args:
        variables (List[str]): a list of variables to be recoded (specified by user)
        mappings (dict): a dictionary of mappings from old to new encoding

    Returns:
        void
    """

    def __init__(self, variables, mappings):

        # Error handling: check to ensure variables is a list
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')

        # Error handling: check to ensure variables is a dict
        if not isinstance(mappings, dict):
            raise ValueError('mapping should be a dictionary')

        # set attributes at instantiation of class
        self.variables = variables
        self.mappings = mappings

    def fit(self, X,
            y=None):  # need to have y as argument to make class compatible with sklearn pipeline
        """ Fit

        Args:
            X (DataFrame): a input dataframe of features to train the transformer
            y (DataFrame): a input Series of response variable to train the transformer (optional)

        Returns:
            self
        """
        # We don't need to learn any parameters for this transformer. Nonetheless, we still need
        # to include a fit method so that the Transformer class would be compatible to sklearn

        return self

    def transform(self, X):
        """ Transform

        Args:
            X (DataFrame): a input dataframe of features to be transformed

        Returns:
            X (DataFrame): the transformed Dataframe of features
        """

        # Make a copy of the input Dataframe of features to be transformed
        # so we won't overwrite the original Dataframe sthat was passed as argument
        X = X.copy()

        # Perform recoding of the levels of var
        for var in self.variables:
            X[var] = X[var].replace(self.mappings)

        return X
    


# This custom step saves the column names of its input dataframe as a binary .pkl file
class save_col_names(BaseEstimator, TransformerMixin):
    """
    Constructor

    Args:
        step_name: the name of the step
        prior to "save_col_names". I want to
        save the column names of that step

    Returns:
        void
    """

    def __init__(self):
        pass

    def fit(
        self, X, y=None
    ):  # need to have y as argument to make class compatible with sklearn pipeline
        """Fit

        Args:
            X (DataFrame): a input dataframe of features
            to train the transformer
            y (DataFrame): a input Series of response variable
            to train the transformer (optional)

        Returns:
            self
        """

        return self

    def transform(self, X):
        """Transform

        Args:
            X (DataFrame): a input dataframe of features to be transformed

        """

        # Make a copy of the input dataframe
        # so we won't overwrite the original Dataframe that was passed as argument
        X = X.copy()

        # Get feature names
        feature_names = X.columns

        # Save as pickle
        with open(
            Path("outputs","processed_col_names.pkl"), "wb"
        ) as f:
            pickle.dump(feature_names, f)

        return X



