from sklearn.datasets import load_breast_cancer
from tabpfn_extensions import TabPFNClassifier
from sklearn.model_selection import train_test_split
from tabpfn_extensions.interpretability.pdp import partial_dependence_plots
import matplotlib.pyplot as plt

data = load_breast_cancer()
X, y = data.data, data.target
X = X[:30]
y = y[:30]
feature_names = list(data.feature_names)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# Initialize and train model
clf = TabPFNClassifier()
clf.fit(X_train, y_train)

# 1D PD for the first 3 features + a 2D interaction plot
disp = partial_dependence_plots(
    estimator=clf,
    X=X_test,
    features=[0, 1, 2, (0, 3)],
    grid_resolution=30,
    kind="average",      # try "individual" to see ICE curves
    target_class=1,      # positive class if using predict_proba
)
disp.figure_.suptitle("Partial dependence")

plt.show()
plt.savefig("pdp.png")
