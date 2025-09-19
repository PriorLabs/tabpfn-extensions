from __future__ import annotations

import numpy as np
from tabpfn_common_utils.telemetry import set_extension

from tabpfn_extensions.utils import TabPFNClassifier, TabPFNRegressor


@set_extension("embedding")
class TabPFNEmbedding:
    """TabPFNEmbedding is a utility for extracting embeddings from TabPFNClassifier or TabPFNRegressor models.
    It supports standard training (vanilla embedding) as well as K-fold cross-validation for embedding extraction.

    - When `n_fold=0`, the model extracts vanilla embeddings by training on the entire dataset.
    - When `n_fold>0`, K-fold cross-validation is applied based on the method proposed in
      "A Closer Look at TabPFN v2: Strength, Limitation, and Extension" (https://arxiv.org/abs/2502.17361),
      where a larger `n_fold` improves embedding effectiveness.

    NOTE: This functionality requires the full TabPFN implementation (pip install tabpfn)
    and is not compatible with the TabPFN client (pip install tabpfn-client). The client
    version does not provide access to model embeddings.

    Parameters:
        tabpfn_clf : TabPFNClassifier, optional
            An instance of TabPFNClassifier to handle classification tasks.
        tabpfn_reg : TabPFNRegressor, optional
            An instance of TabPFNRegressor to handle regression tasks.
        n_fold : int, default=0
            Number of folds for K-fold cross-validation. If set to 0, standard training is used.

    Attributes:
        model : TabPFNClassifier or TabPFNRegressor
            The model used for embedding extraction.

    Examples:
    ```python
    >>> from tabpfn_extensions import TabPFNClassifier  # Must use full TabPFN package
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.datasets import fetch_openml
    >>> X, y = fetch_openml(name='kc1', version=1, as_frame=False, return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    >>> clf = TabPFNClassifier(n_estimators=1)
    >>> embedding_extractor = TabPFNEmbedding(tabpfn_clf=clf, n_fold=0)
    >>> train_embeddings = embedding_extractor.get_embeddings(X_train, y_train, X_test, data_source="train")
    >>> test_embeddings = embedding_extractor.get_embeddings(X_train, y_train, X_test, data_source="test")
    ```
    """

    def __init__(
        self,
        tabpfn_clf: TabPFNClassifier | None = None,
        tabpfn_reg: TabPFNRegressor | None = None,
        n_fold: int = 0,
    ) -> None:
        """Initializes the TabPFNEmbedding instance.

        Args:
            tabpfn_clf (Optional[TabPFNClassifier]): An instance of TabPFN classifier (if available).
            tabpfn_reg (Optional[TabPFNRegressor]): An instance of TabPFN regressor (if available).
            n_fold (int): Number of folds for cross-validation. If 0, cross-validation is not used.

        Raises:
            ImportError: If using TabPFN client implementation which doesn't support embeddings.
        """
        self.tabpfn_clf = tabpfn_clf
        self.tabpfn_reg = tabpfn_reg
        self.model = self.tabpfn_clf if self.tabpfn_clf is not None else self.tabpfn_reg
        self.n_fold = n_fold

        # Check if the model is the full TabPFN implementation, not the client
        if self.model is not None:
            if "tabpfn_client" in str(self.model.__class__.__module__):
                raise ImportError(
                    "TabPFNEmbedding requires the full TabPFN implementation (pip install tabpfn). "
                    "The TabPFN client (pip install tabpfn-client) does not support embedding extraction.",
                )

            # Verify the model has a get_embeddings method
            if not hasattr(self.model, "get_embeddings"):
                raise AttributeError(
                    f"The provided model of type {type(self.model)} does not have a get_embeddings method. "
                    "Make sure you're using the full TabPFN implementation (pip install tabpfn).",
                )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Trains the TabPFN model on the given dataset.

        Args:
            X_train (np.ndarray): Training feature data.
            y_train (np.ndarray): Training target labels.

        Raises:
            ValueError: If no model is set before calling fit.
        """
        if self.model is None:
            raise ValueError("No model has been set.")
        self.model.fit(X_train, y_train)

    def get_embeddings(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X: np.ndarray,
        data_source: str,
    ) -> np.ndarray:
        """Extracts embeddings for the given dataset using the trained model.

        Args:
            X_train (np.ndarray): Training feature data.
            y_train (np.ndarray): Training target labels.
            X (np.ndarray): Data for which embeddings are to be extracted.
            data_source (str): Specifies the data source ("test" for test data).

        Returns:
            np.ndarray: The extracted embeddings.

        Raises:
            ValueError: If no model is set before calling get_embeddings.

        """
        if self.model is None:
            raise ValueError("No model has been set.")

        # If no cross-validation is used, train and return embeddings directly

        if self.n_fold == 0:
            self.model.fit(X_train, y_train)
            return self.model.get_embeddings(X, data_source=data_source)
        elif self.n_fold >= 2:
            if data_source == "test":
                self.model.fit(X_train, y_train)
                return self.model.get_embeddings(X, data_source=data_source)
            else:
                from sklearn.model_selection import KFold

                kf = KFold(n_splits=self.n_fold, shuffle=False)
                embeddings = []
                for train_index, val_index in kf.split(X_train):
                    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
                    y_train_fold, _y_val_fold = y_train[train_index], y_train[val_index]
                    self.model.fit(X_train_fold, y_train_fold)
                    embeddings.append(
                        self.model.get_embeddings(X_val_fold, data_source="test"),
                    )
                return np.concatenate(embeddings, axis=1)
        else:
            raise ValueError("n_fold must be greater than 1.")
