"""TabPFN Distillation Example

This example shows how to use TabPFN as a teacher model for distillation in AutoGluon.

NOTE: You may need to install packages to support student models.
"""

import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import (
    AutoTabPFNClassifier,
)


def run_distillation_example(
    train_student_without_distillation: bool = False,
    use_cross_validation: bool = False,
    n_splits: int = 5,
):
    """An example of using TabPFN as a teacher model for distillation in AutoGluon."""
    # 1. Load data from OpenML
    data_id = 866  # Using the 'credit-g' dataset as an example
    X, y = fetch_openml(
        data_id=data_id, as_frame=True, parser="auto", return_X_y=True
    )

    # Cap dataset size for a faster run, and ensure X and y are pandas objects
    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
        raise TypeError("X and y are not pandas objects as expected.")
    # X = X.head(200)
    # y = y.head(200)

    # Get the label column name from the target series
    if not isinstance(y.name, str):
        raise TypeError(f"y.name should be a string, got {type(y.name)}")
    label_col = y.name

    # Preprocess the target variable from string to integers
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y), index=y.index, name=y.name)

    if not use_cross_validation:
        # Original logic with a single train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)
        y_train = pd.Series(y_train)
        y_test = pd.Series(y_test)

        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)

        train_data = TabularDataset(train_data)
        test_data = TabularDataset(test_data)

        print("🚀 Training the Teacher Model (AutoTabPFNClassifier)...")
        teacher_model = AutoTabPFNClassifier(
            n_ensemble_models=1,
            n_estimators=1,
            max_time=180,
            phe_fit_args={"num_bag_folds": 0, "num_stack_levels": 0},
        )
        teacher_model.fit(X_train, y_train)
        teacher_predictor = teacher_model.predictor_

        test_data = test_data.rename(columns={label_col: "_target_"})
        teacher_predictor.leaderboard(test_data)

        print("\n🎓 Distilling knowledge to a student model...")
        train_data = train_data.rename(columns={label_col: "_target_"})
        student_hyperparameters = {"GBM": {}, "CAT": {}, "RF": {}}
        teacher_predictor.distill(
            train_data=train_data,
            hyperparameters=student_hyperparameters,
            time_limit=180,
        )

        print("\n📊 Evaluating the student model...")
        student_leaderboard = teacher_predictor.leaderboard(test_data)
        print(student_leaderboard)

        if train_student_without_distillation:
            print("\n🏋️ Training student models without distillation for comparison...")
            student_predictor_no_distill = TabularPredictor(
                label="_target_", path="AutogluonModels/student_no_distill"
            )
            student_predictor_no_distill.fit(
                train_data,
                hyperparameters=student_hyperparameters,
                time_limit=180,
            )
            print("\n📊 Evaluating the student model (no distillation)...")
            student_leaderboard_no_distill = student_predictor_no_distill.leaderboard(
                test_data
            )
            print(student_leaderboard_no_distill)

            student_leaderboard["training_method"] = "distillation"
            student_leaderboard_no_distill["training_method"] = "direct"
            combined_leaderboard = pd.concat(
                [student_leaderboard, student_leaderboard_no_distill]
            )
            output_filename = "leaderboard_comparison_new.csv"
            combined_leaderboard.to_csv(output_filename, index=False)
        else:
            output_filename = "leaderboard.csv"
            student_leaderboard.to_csv(output_filename, index=False)

        print("\n🏁 Distillation Example Finished 🏁")
        print(f"Leaderboard(s) saved to {output_filename}")

    else:
        # Cross-validation logic
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        all_leaderboards = []

        for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
            print(f"\n===== FOLD {fold + 1}/{n_splits} =====")
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            train_data = pd.concat([X_train, y_train], axis=1)
            val_data = pd.concat([X_val, y_val], axis=1)
            train_data = TabularDataset(train_data)
            val_data = TabularDataset(val_data)

            print("🚀 Training the Teacher Model (AutoTabPFNClassifier)...")
            teacher_model = AutoTabPFNClassifier(
                n_ensemble_models=1,
                n_estimators=1,
                max_time=180,
                phe_fit_args={"num_bag_folds": 0, "num_stack_levels": 0},
                phe_init_args={
                    "path": f"AutogluonModels/teacher_fold_{fold}"
                },
            )
            teacher_model.fit(X_train, y_train)
            teacher_predictor = teacher_model.predictor_

            print("\n🎓 Distilling knowledge to a student model...")
            train_data_renamed = train_data.rename(columns={label_col: "_target_"})
            val_data_renamed = val_data.rename(columns={label_col: "_target_"})
            student_hyperparameters = {"GBM": {}, "CAT": {}, "RF": {}}
            teacher_predictor.distill(
                train_data=train_data_renamed,
                hyperparameters=student_hyperparameters,
                time_limit=180,
            )

            print("\n📊 Evaluating the distilled student model...")
            distilled_leaderboard = teacher_predictor.leaderboard(val_data_renamed)
            distilled_leaderboard["fold"] = fold
            distilled_leaderboard["training_method"] = "distillation"
            fold_leaderboards = [distilled_leaderboard]

            if train_student_without_distillation:
                print(
                    "\n🏋️ Training student models without distillation for comparison..."
                )
                student_predictor_no_distill = TabularPredictor(
                    label="_target_",
                    path=f"AutogluonModels/student_no_distill_fold_{fold}",
                )
                student_predictor_no_distill.fit(
                    train_data_renamed,
                    hyperparameters=student_hyperparameters,
                    time_limit=180,
                )
                print("\n📊 Evaluating the student model (no distillation)...")
                no_distill_leaderboard = student_predictor_no_distill.leaderboard(
                    val_data_renamed
                )
                no_distill_leaderboard["fold"] = fold
                no_distill_leaderboard["training_method"] = "direct"
                fold_leaderboards.append(no_distill_leaderboard)

            all_leaderboards.extend(fold_leaderboards)

        final_leaderboard = pd.concat(all_leaderboards, ignore_index=True)
        agg_leaderboard = (
            final_leaderboard.groupby(["model", "training_method"])
            .agg(
                score_val_mean=("score_val", "mean"),
                score_val_std=("score_val", "std"),
                fit_time_mean=("fit_time", "mean"),
                fit_time_std=("fit_time", "std"),
                pred_time_val_mean=("pred_time_val", "mean"),
                pred_time_val_std=("pred_time_val", "std"),
            )
            .reset_index()
        )
        agg_leaderboard = agg_leaderboard.sort_values(
            "score_val_mean", ascending=False
        )

        print("\n===== CROSS-VALIDATION RESULTS (AVG & STD) =====")
        print(agg_leaderboard)
        output_filename = "leaderboard_cv_summary.csv"
        agg_leaderboard.to_csv(output_filename, index=False)
        print(f"CV summary saved to {output_filename}")


if __name__ == "__main__":
    run_distillation_example(
        train_student_without_distillation=True, use_cross_validation=True, n_splits=5
    )
