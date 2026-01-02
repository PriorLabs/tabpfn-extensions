"""Conformal prediction for TabPFN with missing data patterns.

This module provides conformal prediction intervals that are calibrated
for different missing data patterns in the input features.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    # Try standard TabPFN package first
    from tabpfn import TabPFNRegressor
except ImportError:
    # Fall back to TabPFN client
    from tabpfn_client import  TabPFNRegressor


class CP_MDA_TabPFNRegressor:
    """
    Compute the correction terms for missing data masks using conformal prediction.

    Parameters:
        X_train : matrix-like of shape (n_samples, n_predictors)

        Y_train : array-like of continuous outcome with shape (n_samples,)

        quantiles : array with three arguments denoting the quantiles of interest.
            The default is [0.05, 0.5, 0.95], where the first indicates the lower bound,
            the second the median, and the third the upper bound.

        val_size : float between 0 and 1, indicating the size of the validation set
            as a fraction of the training data.


    Returns:
     mask_unique: DataFrame with the correction terms for each mask.

     model: Fitted TabPFNRegressor model.

    """

    def __init__(self, X_train, Y_train, quantiles, val_size, seed):
        self.X = pd.DataFrame(X_train)
        self.Y = Y_train
        self.quantiles = quantiles
        self.val_size = val_size
        self.alpha = quantiles[0] * 2
        self.seed = seed

    def calc_correction_term(self, predictions, y_val, alpha):
        """Calculate the correction term for conformal prediction."""
        # obtain the lowerbound, median, and upperbound
        lb, pred, ub = predictions
        # calculate difference between bounds and observed values
        error_lb = (lb - y_val)
        error_ub = (y_val - ub)
        s = np.maximum(error_lb, error_ub)

        # obtain the emperical quantile
        Q_use = (1 - alpha) * (1 + 1/len(s))
        correction_term = np.quantile(s, Q_use)
        return correction_term

    def split_data(self):
        """Split data into training and validation sets."""
        # create df with missing data indicator
        missing_bool_df = self.X.isnull().astype(int)
        self.X_train, self.X_val, Y_train_arr, Y_val_arr, self.Mask_train, self.Mask_val = train_test_split(
            self.X, self.Y, missing_bool_df, test_size=self.val_size, random_state = self.seed
        )

        # Convert Y arrays back to pandas Series to maintain .iloc functionality
        self.Y_train = pd.Series(Y_train_arr, index=self.X_train.index)
        self.Y_val = pd.Series(Y_val_arr, index=self.X_val.index)

    def run_TABPFN(self):
        """Fit the TabPFN model."""
        # fit model
        m_fit = TabPFNRegressor()
        m_fit.fit(self.X_train, self.Y_train)
        self.model = m_fit

    def mask_preprocess(self):
        """Preprocess masks and identify nested relationships."""
        # drop duplicates masks
        mask_unique = self.Mask_val.drop_duplicates().copy()
        # add mask id
        mask_unique["mask_id"] = range(1, len(mask_unique) + 1)
        # Get mask columns (all columns except mask_id)
        mask_cols = [col for col in mask_unique.columns if col != 'mask_id']

        # Check nesting for all pairs of masks
        results = []
        for i, row_a in mask_unique.iterrows():
            mask_a = row_a[mask_cols].values
            mask_a_id = row_a['mask_id']
            nested_masks = []

            for j, row_b in mask_unique.iterrows():
                if i == j:  # Skip comparing mask with itself
                    continue
                mask_b = row_b[mask_cols].values
                mask_b_id = row_b['mask_id']

                if ((mask_b == 1) & (mask_a == 0)).sum() == 0:
                    nested_masks.append(mask_b_id)

            results.append({
                'mask_id': mask_a_id,
                'nested_masks': nested_masks
            })

        self.mask_unique = mask_unique  
        self.mask_nested = pd.DataFrame(results)

    def create_calibration_sets(self):
        """Create calibration sets for each mask pattern."""
        # obtain list of columns
        mask_cols = list(self.Mask_val.columns.values)

        # Using merge to add the id of the mask
        df_with_ids = self.Mask_val.merge(
            self.mask_unique,  
            on=mask_cols,
            how='left'
        )

        for i in self.mask_unique["mask_id"]:
            # select the nested masks
            nested_masks = self.mask_nested[self.mask_nested["mask_id"] == i]["nested_masks"].values[0]
            
            # add the mask itself
            nested_masks_with_self = nested_masks + [i]  # Create new list instead of append

            # obtain indexes for the rows
            indexes = df_with_ids[df_with_ids["mask_id"].isin(nested_masks_with_self)].index

            # select the validation data based on the indices
            X_val_nested = self.X_val.iloc[indexes]
            Y_val_nested = self.Y_val.iloc[indexes]

            # obtain predictions
            predictions = self.model.predict(
                X_val_nested,
                output_type="quantiles",
                quantiles=self.quantiles
            )

            # calculate correction term
            correction_term = self.calc_correction_term(predictions, Y_val_nested, self.alpha)

            # save the correction term to the mask_unique dataframe
            self.mask_unique.loc[self.mask_unique["mask_id"] == i, "correction_term"] = correction_term
            self.mask_unique.loc[self.mask_unique["mask_id"] == i, "val_size"] =  X_val_nested.shape[0]


        return self.mask_unique, self.model

    def fit(self):
        """Convenience method to run the entire pipeline"""
        self.split_data()
        self.run_TABPFN()
        self.mask_preprocess()
        mask_unique, model = self.create_calibration_sets()

        return mask_unique, model

class CP_MDA_TabPFNRegressor_newdata:
    """
    Compute the correction terms for missing data masks using conformal prediction.

    Parameters:

    TabPFN: Fitted TabPFNRegressor model.

    X_new : matrix-like of shape (n_samples, n_predictors)

    quantiles : array with three arguments denoting the quantiles of interest used
                in fitting the model. The default is [0.05, 0.5, 0.95].

    calibration_results : matrix with the correction terms for each mask.


    Returns:
     CP_results: DataFrame with shape (n_samples, 5). Included are the corrected lower bound,
     prediction, corrected upper bound, non-corrected lower bound, and non-corrected upper bound.

    """

    def __init__(self,TabPFN, X_new, quantiles, calibration_results):
        self.TabPFN = TabPFN
        self.X = pd.DataFrame(X_new)
        self.quantiles = quantiles
        self.calibration_results = calibration_results

    def obtain_preds(self):
        """Obtain predictions from fitted model."""
        preds_test = self.TabPFN.predict(
            self.X,
            output_type="quantiles",
            quantiles=self.quantiles
        )
        self.preds_test = preds_test

    def match_mask(self):
      """Add correction terms to the new masks from the test set."""
      mask_test = self.X.isnull().astype(int)
      mask_cols = list(mask_test.columns.values)

      mask_test_cor = mask_test.merge(
            self.calibration_results,
            on=mask_cols,
            how='left'
        )

      # check if there are masks in the test set that are not in the calibration set
      new_masks = mask_test_cor[mask_test_cor["correction_term"].isnull()][mask_cols]

      if new_masks.shape[0] > 0:
          warnings.warn(
              "The following masks are not in the calibration set:\n"
              f"{new_masks.to_string()}\n"
              "The baseline quantile estimates will be returned for those cases."
          )

      self.mask_test_cor = mask_test_cor

    def perform_correction(self):
      """Apply correction terms to the prediction intervals."""
      preds_test = self.preds_test.copy()
      lb_corr = preds_test[0] - self.mask_test_cor["correction_term"].values
      ub_corr = preds_test[2] + self.mask_test_cor["correction_term"].values

      return lb_corr, preds_test[1], ub_corr, preds_test[0], preds_test[2]

    def fit(self):
        """Convenience method to run the entire pipeline"""
        self.obtain_preds()
        self.match_mask()
        CP_results =  self.perf_correction()
        return CP_results
