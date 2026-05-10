import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from tabpfn_extensions import TabPFNClassifier
from tabpfn_extensions.interpretability.pdp import partial_dependence_plots

# Load example dataset
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = list(data.feature_names)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# Initialize and train model. Enable the KV cache (improved with TabPFN-3):
# PDP issues many predicts (grid_resolution per feature * #features) against
# the same fitted model, so caching the encoder pass over X_train avoids
# redoing it on every grid point. The tabpfn-client backend does not
# currently support the KV cache, so fall back to a default constructor.
try:
    clf = TabPFNClassifier(fit_mode="fit_with_cache")
except TypeError:
    clf = TabPFNClassifier()
clf.fit(X_train, y_train)
if hasattr(clf, "executor_"):
    clf.executor_.keep_cache_on_device = True

# 1D PD for the first 3 features + a 2D interaction plot
disp = partial_dependence_plots(
    estimator=clf,
    X=X_test,
    features=[0, 1, 2, (0, 3)],
    grid_resolution=30,
    kind="average",
    target_class=1,
)
disp.figure_.suptitle("Partial dependence")

plt.savefig("pdp_plot.png")
plt.show()
