```python
import numpy as np
from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel

class GenerateSyntheticDataExperiment:
    def __init__(self, task_type):
        self.task_type = task_type

    def run(self, tabpfn, X, y, attribute_names, temp, n_samples, categorical_features=None):
        if categorical_features is None:
            categorical_features = [i for i, feature in enumerate(attribute_names) if isinstance(X[:, i].dtype, np.dtype('int'))]
        # ... rest of the method remains the same ...
