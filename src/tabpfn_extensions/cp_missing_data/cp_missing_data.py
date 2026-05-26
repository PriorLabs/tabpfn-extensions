"""Conformal prediction for TabPFN with missing data patterns.

This module provides conformal prediction intervals that are calibrated
for different missing data patterns in the input features.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from tabpfn_extensions.utils import TabPFNRegressor

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


class CPMDATabPFNRegressor(BaseEstimator, RegressorMixin):
    """Compute the correction terms for missing data masks using conformal prediction.

    Parameters:
        quantiles : list of float, default=[0.05, 0.5, 0.95]
            Three quantiles [lower, median, upper]. Must be symmetric
            (lower + upper == 1).
        val_size : float, default=0.3
            Fraction of training data used for conformal calibration.
        seed : int or None, default=None
            Random seed for the train/calibration split.

    Attributes:
        model_ : TabPFNRegressor
            Fitted TabPFN model.
        calibration_results_ : pd.DataFrame
            Correction terms for each observed missing-data mask.
        alpha_ : float
            Miscoverage level derived from quantiles (lower * 2).
        feature_names_in_ : list
            Column names seen during fit.
    """

    def __init__(
        self,
        quantiles: list[float] | None = None,
        val_size: float = 0.3,
        seed: int | None = None,
    ) -> None:
        self.quantiles = quantiles
        self.val_size = val_size
        self.seed = seed

    def _validate_quantiles(self, quantiles: list[float]) -> None:
        if len(quantiles) != 3:
            raise ValueError(
                f"quantiles must have exactly 3 elements [lower, median, upper], "
                f"got {len(quantiles)}."
            )
        if quantiles[0] >= quantiles[1] or quantiles[1] >= quantiles[2]:
            raise ValueError(f"quantiles must be strictly increasing, got {quantiles}.")
        if not np.isclose(quantiles[0], 1 - quantiles[2]):
            raise ValueError(
                f"quantiles must be symmetric (lower + (1-upper) == 0), "
                f"got {quantiles[0]} - (1 - {quantiles[2]}) = {quantiles[0] - (1 - quantiles[2])}."
            )

    def _calc_correction_term(
        self,
        predictions: tuple[np.ndarray, np.ndarray, np.ndarray],
        y_val: pd.Series,
    ) -> float:
        """Calculate the correction term for conformal prediction."""
        # obtain the lower bound, median, and upper bound
        lb, _, ub = predictions
        # calculate difference between bounds and observed values
        error_lb = lb - y_val
        error_ub = y_val - ub
        s = np.maximum(error_lb, error_ub)

        # obtain the empirical quantile
        Q_use = (1 - self.alpha_) * (1 + 1 / len(s))

        # Check if Q_use is not larger than 1
        if Q_use > 1:
            Q_use = 1
            warnings.warn("Some masks have very small calibration sets", stacklevel=2)

        correction_term = np.quantile(s, Q_use)
        return correction_term

    def _split_data(
        self, x: pd.DataFrame, y: np.ndarray
    ) -> tuple[
        pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame
    ]:
        """Split data into training and validation sets."""
        # create df with missing data indicator
        missing_bool_df = x.isna().astype(int)
        x_train, x_val, y_train_arr, y_val_arr, mask_train, mask_val = train_test_split(
            x, y, missing_bool_df, test_size=self.val_size, random_state=self.seed
        )

        # Convert y arrays back to pandas Series to maintain .iloc functionality
        y_train = pd.Series(y_train_arr, index=x_train.index)
        y_val = pd.Series(y_val_arr, index=x_val.index)

        return x_train, x_val, y_train, y_val, mask_train, mask_val

    def _run_tabpfn(self, x_train: pd.DataFrame, y_train: pd.Series) -> TabPFNRegressor:
        """Fit the TabPFN model."""
        # fit model
        model = TabPFNRegressor()
        model.fit(x_train, y_train)
        return model

    def _mask_preprocess(
        self, mask_val: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Preprocess masks and identify nested relationships."""
        # drop duplicates masks
        mask_unique = mask_val.drop_duplicates().copy()
        # add mask id
        mask_unique["mask_id"] = range(1, len(mask_unique) + 1)
        # Get mask columns (all columns except mask_id)
        mask_cols = [col for col in mask_unique.columns if col != "mask_id"]

        # Check nesting for all pairs of masks
        results = []
        for i, row_a in mask_unique.iterrows():
            mask_a = row_a[mask_cols].values
            mask_a_id = row_a["mask_id"]
            nested_masks = []

            for j, row_b in mask_unique.iterrows():
                if i == j:  # Skip comparing mask with itself
                    continue
                mask_b = row_b[mask_cols].values
                mask_b_id = row_b["mask_id"]

                if ((mask_b == 1) & (mask_a == 0)).sum() == 0:
                    nested_masks.append(mask_b_id)

            results.append({"mask_id": mask_a_id, "nested_masks": nested_masks})

        mask_nested = pd.DataFrame(results)
        return mask_unique, mask_nested

    def _create_calibration_sets(
        self,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        mask_val: pd.DataFrame,
        mask_unique: pd.DataFrame,
        mask_nested: pd.DataFrame,
        model: TabPFNRegressor,
    ) -> tuple[pd.DataFrame, TabPFNRegressor]:
        """Create calibration sets for each mask pattern."""
        # obtain list of columns
        mask_cols = list(mask_val.columns.values)

        # Using merge to add the id of the mask
        # use original index values
        df_with_ids = mask_val.reset_index().merge(
            mask_unique, on=mask_cols, how="left"
        )

        for i in mask_unique["mask_id"]:
            # select the nested masks
            nested_masks = mask_nested[mask_nested["mask_id"] == i][
                "nested_masks"
            ].values[0]

            # add the mask itself
            nested_masks_with_self = [*nested_masks, i]

            # obtain indexes for the rows
            indexes = df_with_ids[df_with_ids["mask_id"].isin(nested_masks_with_self)][
                "index"
            ]

            # select the validation data based on the indices
            x_val_nested = x_val.loc[indexes].copy()
            y_val_nested = y_val.loc[indexes].copy()

            # SET ENTIRE COLUMNS TO NaN WHERE THE MASK HAS MISSING VALUES
            current_mask = mask_unique[mask_unique["mask_id"] == i][mask_cols].iloc[0]

            # For each column where the mask indicates missing (value = 1), set entire column to NaN
            for col_idx, col_name in enumerate(mask_cols):
                if current_mask.iloc[col_idx] == 1:
                    x_val_nested.loc[:, col_name] = np.nan

            # obtain predictions
            predictions = model.predict(
                x_val_nested, output_type="quantiles", quantiles=self.quantiles_
            )

            # calculate correction term
            correction_term = self._calc_correction_term(predictions, y_val_nested)

            # save the correction term to the mask_unique dataframe
            mask_unique.loc[mask_unique["mask_id"] == i, "correction_term"] = (
                correction_term
            )
            mask_unique.loc[mask_unique["mask_id"] == i, "val_size"] = (
                x_val_nested.shape[0]
            )

        return mask_unique, model

    def fit(self, x_train: ArrayLike, y_train: ArrayLike) -> "CPMDATabPFNRegressor":
        """Fit the model and compute conformal calibration corrections.

        Parameters:
            x_train : matrix-like of shape (n_samples, n_predictors)
            y_train : array-like of continuous outcome with shape (n_samples,)
        """

        # check if quantiles are correct
        quantiles = self.quantiles if self.quantiles is not None else [0.05, 0.5, 0.95]
        self._validate_quantiles(quantiles)
        self.quantiles_ = quantiles
        self.alpha_ = self.quantiles_[0] * 2

        # Store and parse the data
        x = pd.DataFrame(x_train)
        y = y_train

        # save colnames for check later
        self.feature_names_in_ = list(x.columns)

        # Run through all the functions
        x_train_split, x_val, y_train_split, y_val, _, mask_val = self._split_data(x, y)
        model = self._run_tabpfn(x_train_split, y_train_split)
        mask_unique, mask_nested = self._mask_preprocess(mask_val)
        self.calibration_results_, self.model_ = self._create_calibration_sets(
            x_val, y_val, mask_val, mask_unique, mask_nested, model
        )

        return self

    def _obtain_preds(self, x: pd.DataFrame) -> np.ndarray:
        """Obtain predictions from fitted model."""
        preds = self.model_.predict(
            x, output_type="quantiles", quantiles=self.quantiles_
        )
        return preds

    def _match_mask(self, x: pd.DataFrame) -> pd.DataFrame:
        """Add correction terms to the new masks from the test set."""
        mask_test = x.isna().astype(int)
        mask_cols = list(mask_test.columns.values)

        mask_test_cor = mask_test.merge(
            self.calibration_results_, on=mask_cols, how="left"
        )

        # check if there are masks in the test set that are not in the calibration set
        new_masks = mask_test_cor[mask_test_cor["correction_term"].isna()][mask_cols]

        if new_masks.shape[0] > 0:
            warnings.warn(
                "The following masks are not in the calibration set:\n"
                f"{new_masks.to_string()}\n"
                "The baseline quantile estimates will be returned for those cases.",
                stacklevel=2,
            )

        # Fill NA to return original intervals
        mask_test_cor["correction_term"] = mask_test_cor["correction_term"].fillna(0)

        return mask_test_cor

    def _perform_correction(
        self, preds: np.ndarray, mask_test_cor: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply correction terms to the prediction intervals."""
        lb_corr = preds[0] - mask_test_cor["correction_term"].values
        ub_corr = preds[2] + mask_test_cor["correction_term"].values

        return lb_corr, preds[1], ub_corr, preds[0], preds[2]

    def predict(
        self, x_new: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Obtain predictions for new data with conformalised uncertainty estimates.

        Parameters:
            x_new : matrix-like of shape (n_samples, n_predictors)
        """
        check_is_fitted(self)
        x = pd.DataFrame(x_new)

        # check if new data has the same columns as old data
        if isinstance(x_new, pd.DataFrame):
            if list(x.columns) != self.feature_names_in_:
                raise ValueError(
                    f"Column names of x_new do not match those seen during fit. "
                    f"Expected {self.feature_names_in_}, got {list(x.columns)}."
                )

        preds = self._obtain_preds(x)
        mask_test_cor = self._match_mask(x)
        cp_results = self._perform_correction(preds, mask_test_cor)

        return cp_results
