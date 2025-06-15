#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, StratifiedKFold

logger = logging.getLogger(__name__)


def _save_stratified_splits(
    _splitter: StratifiedKFold | RepeatedStratifiedKFold,
    x: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    auto_fix_stratified_splits: bool = False,
) -> list[list[list[int], list[int]]]:
    """Fix from AutoGluon to avoid unsafe splits for classification if less than n_splits instances exist for all classes.

    https://github.com/autogluon/autogluon/blob/0ab001a1193869a88f7af846723d23245781a1ac/core/src/autogluon/core/utils/utils.py#L70.
    """
    try:
        splits = [
            [train_index, test_index]
            for train_index, test_index in _splitter.split(x, y)
        ]
    except ValueError as e:
        x = pd.DataFrame(x)
        y = pd.Series(y)
        y_dummy = pd.concat([y, pd.Series([-1] * n_splits)], ignore_index=True)
        X_dummy = pd.concat([x, x.head(n_splits)], ignore_index=True)
        invalid_index = set(y_dummy.tail(n_splits).index)
        splits = [
            [train_index, test_index]
            for train_index, test_index in _splitter.split(X_dummy, y_dummy)
        ]
        len_out = len(splits)
        for i in range(len_out):
            train_index, test_index = splits[i]
            splits[i][0] = [
                index for index in train_index if index not in invalid_index
            ]
            splits[i][1] = [index for index in test_index if index not in invalid_index]

        # only rais afterward because only now we know that we cannot fix it
        if not auto_fix_stratified_splits:
            raise AssertionError(
                "Cannot split data in a stratifed way with each class in each subset of the data.",
            ) from e
    except UserWarning as e:
        # Assume UserWarning for not enough classes for correct stratified splitting.
        raise e

    return splits


def fix_split_by_dropping_classes(
    x: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    spliter_kwargs: dict,
) -> list[list[list[int], list[int]]]:
    """Fixes stratifed splits for edge case.

    For each class that has fewer instances than number of splits, we oversample before split to n_splits and then remove all oversamples and
    original samples from the splits; effectively removing the class from the data without touching the indices.
    """
    val, counts = np.unique(y, return_counts=True)
    too_low = val[counts < n_splits]
    too_low_counts = counts[counts < n_splits]

    y_dummy = pd.Series(y.copy())
    X_dummy = pd.DataFrame(x.copy())
    org_index_max = len(X_dummy)
    invalid_index = []

    assert len(too_low) == len(
        too_low_counts,
    ), "too_low and too_low_counts must have the same length!"
    for c_val, c_count in zip(too_low, too_low_counts):
        fill_missing = n_splits - c_count
        invalid_index.extend(np.where(y == c_val)[0])
        y_dummy = pd.concat(
            [y_dummy, pd.Series([c_val] * fill_missing)],
            ignore_index=True,
        )
        X_dummy = pd.concat(
            [X_dummy, pd.DataFrame(x).head(fill_missing)],
            ignore_index=True,
        )

    invalid_index.extend(list(range(org_index_max, len(y_dummy))))
    splits = _save_stratified_splits(
        _splitter=StratifiedKFold(**spliter_kwargs),
        x=X_dummy,
        y=y_dummy,
        n_splits=n_splits,
    )
    len_out = len(splits)
    for i in range(len_out):
        train_index, test_index = splits[i]
        splits[i][0] = [index for index in train_index if index not in invalid_index]
        splits[i][1] = [index for index in test_index if index not in invalid_index]

    return splits


def assert_valid_splits(
    splits: list[list[list[int], list[int]]],
    y: np.ndarray,
    *,
    non_empty: bool = True,
    each_selected_class_in_each_split_subset: bool = True,
    same_length_training_splits: bool = True,
):
    """Verify that the splits are valid."""
    if non_empty:
        assert len(splits) != 0, "No splits generated!"
        for split in splits:
            assert len(split) != 0, "Some split is empty!"
            assert len(split[0]) != 0, "A train subset of a split is empty!"
            assert len(split[1]) != 0, "A test subset of a split is empty!"

    if each_selected_class_in_each_split_subset:
        # As we might drop classes, we first need to build the set of classes that are in the splits.
        #   - 2nd unique is for speed up purposes only.
        _real_y = set(
            np.unique([c for split in splits for c in np.unique(y[split[1]])]),
        )
        # Now we need to check that each class that exists in all splits is in each split.
        for split in splits:
            assert _real_y == (
                set(np.unique(y[split[0]]))
            ), "A class is missing in a train subset!"
            assert _real_y == (
                set(np.unique(y[split[1]]))
            ), "A class is missing in a test subset!"

    if same_length_training_splits:
        for split in splits:
            assert len(split[0]) == len(
                splits[0][0],
            ), "A train split has different amount of samples!"


def _equalize_training_splits(
    input_splits: list[list[list[int], list[int]]],
    rng: np.random.RandomState,
) -> list[list[list[int], list[int]]]:
    """Equalize training splits by duplicating samples in too small splits."""
    splits = input_splits[:]
    n_max_train_samples = max(len(split[0]) for split in splits)
    for split in splits:
        curr_split_len = len(split[0])
        if curr_split_len < n_max_train_samples:
            missing_samples = n_max_train_samples - curr_split_len
            split[0].extend(
                [int(dup_i) for dup_i in rng.choice(split[0], size=missing_samples)],
            )
            split[0] = sorted(split[0])

    return splits


def get_cv_split_for_data(
    x: np.ndarray,
    y: np.ndarray,
    splits_seed: int,
    n_splits: int,
    *,
    stratified_split: bool,
    safety_shuffle: bool = True,
    auto_fix_stratified_splits: bool = False,
    force_same_length_training_splits: bool = False,
) -> list[list[list[int], list[int]]] | str:
    """Safety shuffle and generate (safe) splits.

    If it returns str at the first entry, no valid split could be generated and the str is the reason why.
    Due to the safety shuffle, the original x and y are also returned and must be used.

    Note: the function does not support repeated splits at this point.
    Simply call this function multiple times with different seeds to get repeated splits.

    Test with:

    ```python
        if __name__ == "__main__":
        print(
            get_cv_split_for_data(
                x=np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).T,
                y=np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4]),
                splits_seed=42,
                n_splits=3,
                stratified_split=True,
                auto_fix_stratified_splits=True,
            )
        )
    ```

    Args:
        x: The data to split.
        y: The labels to split.
        splits_seed: The seed to use for the splits. Or a RandomState object.
        n_splits: The number of splits to generate.
        stratified_split: Whether to use stratified splits.
        safety_shuffle: Whether to shuffle the data before splitting.
        auto_fix_stratified_splits: Whether to try to fix stratified splits automatically.
            Fix by dropping classes with less than n_splits samples.
        force_same_length_training_splits: Whether to force the training splits to have the same amount of samples.
            Force by duplicating random instance in the training subset of a too small split until all training splits have the same amount of samples.
    Out:
        A list of pairs of indexes, where in each pair first come the train examples, then test. So we get something like
        `[[TRAIN_INDICES_0, TEST_INDICES_0], [TRAIN_INDICES_1, TRAIN_INDICES_1]]` for 2 splits.
        Or a string if no valid split could be generated whereby the string gives the reason.
    """
    assert len(x) == len(y), "x and y must have the same length!"

    rng = np.random.RandomState(splits_seed)
    if safety_shuffle:
        p = rng.permutation(len(x))
        x, y = x[p], y[p]
    spliter_kwargs = {"n_splits": n_splits, "shuffle": True, "random_state": rng}

    if not stratified_split:
        splits = [list(tpl) for tpl in KFold(**spliter_kwargs).split(x, y)]
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                splits = _save_stratified_splits(
                    _splitter=StratifiedKFold(**spliter_kwargs),
                    x=x,
                    y=y,
                    n_splits=n_splits,
                    auto_fix_stratified_splits=auto_fix_stratified_splits,
                )
            except UserWarning as e:
                logger.debug(e)
                if auto_fix_stratified_splits:
                    logger.debug("Trying to fix stratified splits automatically...")
                    splits = fix_split_by_dropping_classes(
                        x=x,
                        y=y,
                        n_splits=n_splits,
                        spliter_kwargs=spliter_kwargs,
                    )
                else:
                    splits = "Cannot generate valid stratified splits for dataset without losing classes in some subsets!"
            except AssertionError as e:
                logger.debug(e)
                splits = "Cannot generate valid stratified splits for dataset without losing classes in some subsets!"

    if isinstance(splits, str):
        return splits

    if force_same_length_training_splits:
        splits = _equalize_training_splits(splits, rng)

    assert_valid_splits(
        splits=splits,
        y=y,
        non_empty=True,
        same_length_training_splits=force_same_length_training_splits,
        each_selected_class_in_each_split_subset=stratified_split,
    )

    if safety_shuffle:
        # Revert to correct outer scope indices
        for split in splits:
            split[0] = sorted(p[split[0]])
            split[1] = sorted(p[split[1]])

    return splits
