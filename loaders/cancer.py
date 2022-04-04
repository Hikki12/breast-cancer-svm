import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def normalize_vector(x):
    """Normalizes a vector.

    Args:
        x: input data
    """
    x_min = x.min()
    x_max = x.max()
    xn = (x - x_min) / (x_max - x_min)
    return xn


def load_cancer_data(split: bool = True, test_size: float = .2, random_state: int = 42, 
    console: bool = True, normalize: bool = True) -> tuple:
    """Loads splitted data for test and train.
    
    Args:
        split: split data?
        test_size: percentage of data to be used as test 
        random_state: random seed
        console: display dataset info?
        normalize: normalize input data?
               
    Returns:
        X_train: input data for train
        X_test: input data for test
        y_train: output data for train
        y_test: output data for test
    """
    cancer = load_breast_cancer()

    df = pd.DataFrame(
        np.c_[cancer["data"], cancer["target"]],
        columns=np.append(cancer["feature_names"], ["target"]),
    )

    X = df.drop(["target"], axis=1)
    y = df["target"]

    if split:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        if normalize:
            X_train = normalize_vector(X_train)
            X_test = normalize_vector(X_test)
    else:
        if normalize:
            X = normalize_vector(X)
            y = normalize_vector(y)

    if console:
        print("======== CANCER DATA LOADED ======= ")
        print()
        print(f"dataset shape: {X.shape} ")
        if split:
            print(f"X train shape: {X_train.shape}")
            print(f"X test shape: {X_test.shape}")
            print(f"y train shape: {y_train.shape}")
            print(f"y test shape: {y_test.shape}")
        print()
        print("----")

    if split:
        return X_train, X_test, y_train, y_test

    return X, y
