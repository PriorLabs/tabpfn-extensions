"""WARNING: This example may run slowly on CPU-only systems.
For better performance, we recommend running with GPU acceleration.
Feature selection involves training multiple TabPFN models, which is computationally intensive.
"""

import os

from sklearn.datasets import load_breast_cancer

from tabpfn_extensions import TabPFNClassifier, interpretability

# Under FAST_TEST_MODE=1 (set by the CI example tests) the workload shrinks so
# the example finishes quickly; results are correspondingly rougher.
FAST_TEST_MODE = os.environ.get("FAST_TEST_MODE") == "1"

# Load data
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names
if FAST_TEST_MODE:
    # Selection cost grows with rows, candidate features, and rounds.
    X, y, feature_names = X[:100, :10], y[:100], feature_names[:10]

# Initialize model. Single estimator keeps the runtime manageable — feature
# selection runs many TabPFN fits per round.
clf = TabPFNClassifier(n_estimators=1)

# Feature selection. With verbose=True (the default) the wrapper prints the
# baseline CV score on all features, the per-round picks, and the selected
# names + CV score on the subset. The same numbers are also available on
# the returned FeatureSelectionResult for programmatic use.
result = interpretability.feature_selection.feature_selection(
    estimator=clf,
    X=X,
    y=y,
    n_features_to_select=2 if FAST_TEST_MODE else 4,
    feature_names=list(feature_names),
)

# `result.selected_names` is populated because we passed `feature_names`.
# `result.selector.transform(X)` would project to just those columns;
# `result.support_mask` / `result.selected_indices` are also available.
print("\nProgrammatic summary:")
print(f"Selected features: {result.selected_names}")
print(
    f"CV score before / after: "
    f"{result.baseline_score_mean:.4f} -> {result.selected_score_mean:.4f}"
)
