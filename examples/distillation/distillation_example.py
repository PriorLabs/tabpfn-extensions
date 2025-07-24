"""TabPFN Distillation Example

This example shows how to use TabPFN as a teacher model for distillation in AutoGluon.

NOTE: You may need to install packages to support student models.
"""

import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import Bunch

from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import (
    AutoTabPFNClassifier,
)


def run_distillation_example():
    """
    An example of using TabPFN as a teacher model for distillation in AutoGluon.
    """
    # 1. Load data from OpenML
    data_id = 31  # Using the 'credit-g' dataset as an example
    bunch = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    if not isinstance(bunch, Bunch):
        raise TypeError(f"Expected a Bunch object, but got {type(bunch)}")

    data: pd.DataFrame = bunch.frame
    data = data.head(200)  # Optional: Cap dataset size for a faster run
    label_col = "class"

    X: pd.DataFrame = pd.DataFrame(data.drop(columns=[label_col]))
    y: pd.Series = pd.Series(data[label_col])

    # Preprocess the target variable from string to integers
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y), index=y.index, name=y.name)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)

    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    # For AutoGluon, we can pass data as a pandas DataFrame
    train_data = TabularDataset(train_data)
    test_data = TabularDataset(test_data)

    # 2. Train the Teacher Model (AutoTabPFNClassifier)
    # The AutoTabPFNClassifier is a scikit-learn compatible model that internally
    # uses AutoGluon to create a powerful ensemble of TabPFN models.
    # We instantiate it directly and fit it on our data.
    print("🚀 Training the Teacher Model (AutoTabPFNClassifier)...")
    teacher_model = AutoTabPFNClassifier(
        n_ensemble_models=1,
        n_estimators=1,
        max_time=180,  # This time limit is for the internal AutoGluon process
        phe_fit_args={
            "num_bag_folds": 0,
            "num_stack_levels": 0,
        },  # Disable bagging and stacking for a faster run
    )
    # The `AutoTabPFNClassifier` is a scikit-learn-style model, so we fit it on the raw data.
    teacher_model.fit(X_train, y_train)

    # The teacher_model now contains a fitted AutoGluon TabularPredictor.
    # We can access it via the .predictor_ attribute.
    teacher_predictor = teacher_model.predictor_

    test_data.rename(columns={"class": "_target_"}, inplace=True)
    teacher_predictor.leaderboard(test_data)

    # 3. Train the Student Model via Distillation
    print("\n🎓 Distilling knowledge to a student model...")

    # The `distill` method trains a new predictor where models learn from the teacher's predictions
    train_data.rename(columns={"class": "_target_"}, inplace=True)
    teacher_predictor.distill(
        train_data=train_data,
        # hyperparameters={"GBM": {}, "CAT": {}, "RF": {}}, # uncomment to use default autogluon models
        time_limit=180,
    )

    # 4. Evaluate and Compare
    print("\n📊 Evaluating the student model...")
    # After distillation, the predictor is updated with the student models.
    # We can now show the leaderboard of the distilled predictor.
    student_leaderboard = teacher_predictor.leaderboard(test_data)
    print(student_leaderboard)

    print("\n🏁 Distillation Example Finished 🏁")
    print(
        "Leaderboard shows the performance of student models trained to mimic the TabPFN teacher."
    )
    print(
        "The '_DSTL' suffix indicates a model trained via distillation."
    )

    # Optional: save leaderboard to csv
    student_leaderboard.to_csv("leaderboard.csv", index=False)


if __name__ == "__main__":
    run_distillation_example()