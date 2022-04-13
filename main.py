from loaders import load_cancer_data
from models import Classifier
import pandas as pd
import numpy as np

# ============================== LOAD DATA =================================

X, y = load_cancer_data(
    split=False, test_size=0.2, random_state=40, normalize=True,
)

# ============================== SETUP MODEL =================================
components = [2*i for i in range(1, 16)]
n_scores = {}
for n_components in components:
    classifier = Classifier(n_components=n_components)
    scores = classifier.train(X, y, cv=5, reduce=False)
    for k, v in scores.items():
        scores[k] = np.mean(v)
    n_scores[f"{n_components}"] = scores

# ============================== RESULTS =================================
results = pd.DataFrame(n_scores)
results.to_csv("./results.csv")