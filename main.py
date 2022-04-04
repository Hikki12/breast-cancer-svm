from loaders import load_cancer_data
from models import Classifier

# ============================== LOAD DATA =================================

X, y = load_cancer_data(
    split=False, test_size=0.2, random_state=40, normalize=True,
)

# ============================== SETUP MODEL =================================

classifier = Classifier(n_components=5)
scores = classifier.train(X, y, cv=5)

# ============================== RESULTS =================================
print(scores)