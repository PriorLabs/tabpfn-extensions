# TabPFN-CRT

This repository provides functionality for feature-level hypothesis testing
using foundation models via the Conditional Randomization Test (CRT).

The implementation treats TabPFN as a fixed predictive model and uses its predictive
log-likelihood to construct a valid test statistic.

The goal is to bridge modern foundation models with classical confirmatory inference,
particularly in settings where p-values and formal hypothesis tests are required.

----------------------------------------------------------------------
Motivation
----------------------------------------------------------------------

Foundation models such as TabPFN are increasingly used in applied research, but their
adoption in hypothesis-driven and feature-selection workflows is often limited by the
absence of principled statistical tests.

Many scientific disciplines still require:
- calibrated p-values
- explicit null hypotheses
- interpretable feature relevance decisions

----------------------------------------------------------------------
When Should You Use TabPFN-CRT?
----------------------------------------------------------------------

Use TabPFN-CRT when you want to test whether a specific feature
provides predictive information about the target beyond other
covariates, while still using a strong foundation model.

For example, this can be useful when:

• validating whether a scientific variable is truly predictive  
• auditing black-box model feature importance  
• performing feature selection with statistically valid p-values  
• working in regulated or high-stakes settings requiring hypothesis tests

Unlike heuristic importance measures (e.g., permutation importance or SHAP),
TabPFN-CRT provides formal hypothesis testing with calibrated p-values.

----------------------------------------------------------------------
Method Overview
----------------------------------------------------------------------

Given data (X, y) and a feature index j, the procedure:

1. Fits a TabPFN model to predict y | X
2. Fits a separate TabPFN model to approximate the conditional distribution X_j | X_-j
3. Computes a test statistic based on the average log predictive density of y
4. Generates a null distribution by resampling X_j from its conditional distribution
5. Computes a right-tailed p-value via the Conditional Randomization Test


Crucially:
- TabPFN is treated as a fixed predictive model
- No Bayesian correctness or uncertainty calibration is assumed
- Validity relies on the CRT framework, not on TabPFN being probabilistically correct

----------------------------------------------------------------------
Basic Usage
----------------------------------------------------------------------

Example 1 — Test a single feature by index:

    from tabpfn_extensions.pval_crt import tabpfn_crt

    result = tabpfn_crt(
        X,
        y,
        j=3,
        B=200,
        alpha=0.05,
    )

    print(result["p_value"], result["reject_null"])


Example 2 — Test a single feature by name (when X is a DataFrame):

    result = tabpfn_crt(
        X,
        y,
        j="age",
    )

    print(result["p_value"])


Example 3 — Test multiple features simultaneously:

    results = tabpfn_crt(
        X,
        y,
        j=["age", "income", "education"],
    )

    for feature, res in results.items():
        print(feature, res["p_value"])


Returned values
---------------

Single feature testing returns a dictionary containing:

- p_value
- reject_null
- observed test statistic
- null distribution
- feature and model metadata

Multiple feature testing returns:

- dict[feature → result dictionary]


----------------------------------------------------------------------
Relation to Prior Work
----------------------------------------------------------------------

This implementation follows the Conditional Randomization Test framework
(e.g., Candès et al., 2018) and adapts it to modern foundation models by:

- using predictive log-likelihood as the test statistic
- allowing flexible conditional modeling via TabPFN
- avoiding retraining or model modification

