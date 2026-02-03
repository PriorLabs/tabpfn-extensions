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

from tabpfn_extensions.utils import TabPFNRegressor

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


class CPMDATabPFNRegressor:
    """Compute the correction terms for missing data masks using conformal prediction.

    Parameters:
        quantiles : array with three arguments denoting the quantiles of interest.
            The default is [0.05, 0.5, 0.95], where the first indicates the lower bound,
            the second the median, and the third the upper bound.
        val_size : float between 0 and 1, indicating the size of the validation set
            as a fraction of the training data.

    Returns:
        mask_unique: DataFrame with the correction terms for each mask.
        model: Fitted TabPFNRegressor model.
    """

    def __init__(
        self,
        quantiles: list[float],
        val_size: float,
        seed: int | None = None
    ) -> None:
        self.quantiles = quantiles
        self.val_size = val_size
        self.alpha = quantiles[0] * 2
        self.seed = seed

    def calc_correction_term(
        self,
        predictions: tuple[np.ndarray, np.ndarray, np.ndarray],
        y_val: pd.Series,
        alpha: float
    ) -> float:
        """Calculate the correction term for conformal prediction."""
        # obtain the lowerbound, median, and upperbound
        lb, pred, ub = predictions
        # calculate difference between bounds and observed values
        error_lb = (lb - y_val)
        error_ub = (y_val - ub)
        s = np.maximum(error_lb, error_ub)

        # obtain the emperical quantile
        Q_use = (1 - alpha) * (1 + 1/len(s))

        # Check is Q_use if not larger then 1
        if Q_use > 1:
            Q_use = 1
            warnings.warn(
                "Some masks have very small calibration sets",  stacklevel=2)

        correction_term = np.quantile(s, Q_use)
        return correction_term

    def split_data(self,
        x: pd.DataFrame,
        y: np.ndarray
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
        """Split data into training and validation sets."""
        # create df with missing data indicator
        missing_bool_df = x.isna().astype(int)
        x_train, x_val, y_train_arr, y_val_arr, mask_train, mask_val = train_test_split(
            x, y, missing_bool_df, test_size=self.val_size, random_state = self.seed
        )

        # Convert y arrays back to pandas Series to maintain .iloc functionality
        y_train = pd.Series(y_train_arr, index=x_train.index)
        y_val = pd.Series(y_val_arr, index=x_val.index)

        return x_train, x_val, y_train, y_val, mask_train, mask_val

    def run_TABPFN(self,
        x_train: pd.DataFrame,
        y_train: pd.Series
    ) -> TabPFNRegressor:
        """Fit the TabPFN model."""
        # fit model
        model = TabPFNRegressor()
        model.fit(x_train, y_train)
        return(model)

    def mask_preprocess(
        self,
        mask_val: pd.DataFrame
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

            results.append({
                "mask_id": mask_a_id,
                "nested_masks": nested_masks
            })

        mask_nested = pd.DataFrame(results)
        return mask_unique, mask_nested

    def create_calibration_sets(
        self,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        mask_val: pd.DataFrame,
        mask_unique: pd.DataFrame,
        mask_nested: pd.DataFrame,
        model: TabPFNRegressor
    ) -> tuple[pd.DataFrame, TabPFNRegressor]:
        """Create calibration sets for each mask pattern."""
        # obtain list of columns
        mask_cols = list(mask_val.columns.values)

        # Using merge to add the id of the mask
        # use original index values
        df_with_ids = mask_val.reset_index().merge(
            mask_unique,
            on=mask_cols,
            how="left"
        )

        for i in mask_unique["mask_id"]:
            # select the nested masks
            nested_masks = mask_nested[mask_nested["mask_id"] == i]["nested_masks"].values[0]

            # add the mask itself
            nested_masks_with_self = [*nested_masks, i]

            # obtain indexes for the rows
            indexes = df_with_ids[df_with_ids["mask_id"].isin(nested_masks_with_self)]["index"]

            # select the validation data based on the indices
            x_val_nested = x_val.loc[indexes]
            y_val_nested = y_val.loc[indexes]

            # SET ENTIRE COLUMNS TO NaN WHERE THE MASK HAS MISSING VALUES
            current_mask = mask_unique[mask_unique["mask_id"] == i][mask_cols].iloc[0]

            # For each column where the mask indicates missing (value = 1), set entire column to NaN
            for col_idx, col_name in enumerate(mask_cols):
                if current_mask.iloc[col_idx] == 1:
                    x_val_nested.loc[:, col_name] = np.nan

            # obtain predictions
            predictions = model.predict(
                x_val_nested,
                output_type="quantiles",
                quantiles=self.quantiles
            )

            # calculate correction term
            correction_term = self.calc_correction_term(predictions, y_val_nested, self.alpha)

            # save the correction term to the mask_unique dataframe
            mask_unique.loc[mask_unique["mask_id"] == i, "correction_term"] = correction_term
            mask_unique.loc[mask_unique["mask_id"] == i, "val_size"] =  x_val_nested.shape[0]

        return mask_unique, model

    def fit(
        self,
        x_train: ArrayLike,
        y_train: ArrayLike
    ) -> tuple[pd.DataFrame, TabPFNRegressor]:
        """Convenience method to run the entire pipeline.

        Parameters:
            x_train : matrix-like of shape (n_samples, n_predictors)
            y_train : array-like of continuous outcome with shape (n_samples,)
        """
        # Store and parse the data
        x = pd.DataFrame(x_train)
        y = y_train

        # Run trough all the functions
        x_train, x_val, y_train, y_val, mask_train, mask_val = self.split_data(x, y)
        model = self.run_TABPFN(x_train, y_train)
        mask_unique, mask_nested = self.mask_preprocess(mask_val)
        mask_unique, model = self.create_calibration_sets(
            x_val, y_val, mask_val, mask_unique, mask_nested, model)

        return mask_unique, model


class CPMDATabPFNRegressorNewData:
    """Compute the correction terms for missing data masks using conformal prediction.

    Parameters:
        tabpfn : Fitted TabPFNRegressor model.
        quantiles : Array with three arguments denoting the quantiles of interest used
            in fitting the model. The default is [0.05, 0.5, 0.95].
        calibration_results : Matrix with the correction terms for each mask.
    """

    def __init__(
        self,
        tabpfn: TabPFNRegressor,
        quantiles: list[float],
        calibration_results: pd.DataFrame
    ) -> None:
        self.tabpfn = tabpfn
        self.quantiles = quantiles
        self.calibration_results = calibration_results

    def obtain_preds(self,
        x: pd.DataFrame) -> np.ndarray:
        """Obtain predictions from fitted model."""
        preds = self.tabpfn.predict(
            x,
            output_type="quantiles",
            quantiles=self.quantiles
        )
        return preds

    def match_mask(self,
        x: pd.DataFrame) -> pd.DataFrame:
        """Add correction terms to the new masks from the test set."""
        mask_test = x.isna().astype(int)
        mask_cols = list(mask_test.columns.values)

        mask_test_cor = mask_test.merge(
                self.calibration_results,
                on=mask_cols,
                how="left"
            )

        # check if there are masks in the test set that are not in the calibration set
        new_masks = mask_test_cor[mask_test_cor["correction_term"].isna()][mask_cols]

        if new_masks.shape[0] > 0:
            warnings.warn(
                "The following masks are not in the calibration set:\n"
                f"{new_masks.to_string()}\n"
                "The baseline quantile estimates will be returned for those cases.",  stacklevel=2
            )

        return mask_test_cor

    def perform_correction(
        self,
        preds: np.ndarray,
        mask_test_cor: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply correction terms to the prediction intervals."""
        lb_corr = preds[0] - mask_test_cor["correction_term"].values
        ub_corr = preds[2] + mask_test_cor["correction_term"].values

        return lb_corr, preds[1], ub_corr, preds[0], preds[2]

    def predict(
        self,
        x_new: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convenience method to run the entire pipeline.

        Parameters:
            x_new : matrix-like of shape (n_samples, n_predictors)
        """
        x = pd.DataFrame(x_new)

        preds = self.obtain_preds(x)
        mask_test_cor = self.match_mask(x)
        cp_results = self.perform_correction(preds, mask_test_cor)

        return cp_results