# Multi-output Prediction Examples

This directory demonstrates how to tackle both multi-output regression and
multi-label classification tasks with TabPFN. The lightweight wrappers
`TabPFNMultiOutputRegressor` and `TabPFNMultiOutputClassifier` automatically
clone base TabPFN models per target so you can scale to multiple outputs with a
single configuration.

The accompanying script shows how to:

1. Generate a synthetic two-target regression dataset with correlated outputs
2. Introduce missing values into the feature matrix to exercise TabPFN's native
   handling of incomplete data
3. Fit the `TabPFNMultiOutputRegressor` with the publicly available v2 model
4. Build a multi-label classification dataset and fit the
   `TabPFNMultiOutputClassifier`
5. Evaluate both tasks to confirm predictions remain finite and performant
