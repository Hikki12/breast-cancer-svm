import joblib
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn import svm
from sklearn.decomposition import PCA
import numpy as np


class Classifier:
    """PCA - SVM classifier."""
    def __init__(self, n_components: int = 0):
        self.svm = svm.SVC()
        self.pca = PCA(n_components=n_components)

    def has_svm(self):
        return self.svm is not None
    
    def has_pca(self):
        return self.pca is not None

    def save_svm(self, filepath: str = "svm.joblib"):
        if self.has_svm():
            self.svm = joblib.dump(self.svm, filepath)

    def save_pca(self, filepath: str = "pca.joblib"):
        if self.has_pca():
            self.pca = joblib.dump(self.pca, filepath)

    def load_svm(self, filepath: str = "svm.joblib"):
        if self.has_model():
            self.svm = joblib.load(filepath)

    def load_pca(self, filepath: str = "pca.joblib"):
        if self.has_model():
            self.pca = joblib.load(filepath)
        
    def reduce_dim(self, X):
        """Applies pca dim reduction.
        
        Args:
            X: input data
        """
        if self.has_pca():
            try:
                X = self.pca.fit_transform(X)
            except Exception as e:
                print("Classiffier:: ", e)
        return X

    def train(self, X, y, cv: int = 5, reduce: bool = True, scoring=None):
        """Trains the model with Cross Validation method.
        Args:
            X: input data
            y: output data
            cv: number of cross validation
            reduce: apply pca?

        Returns:
            scores: list with accuracy scores
        """
        if self.has_svm():
            if reduce:
                X = self.reduce_dim(X)

            scores = cross_validate(
                self.svm, X, y, cv=cv
            )

            return scores

    def predict(self, X, reduce: bool = False):
        """Make a prediction with the model.

        Args:
            X: input data
            reduce: apply reduce?
        """
        if self.has_svm():
            if reduce:
                X = self.reduce_dim(X)
            return self.svm.predict(X)