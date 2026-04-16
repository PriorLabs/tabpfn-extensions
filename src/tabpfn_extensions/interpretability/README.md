# TabPFN Interpretability

## TabPFN Partial Dependence Plots

``partial_dependence_plots`` provides a way to visualize how one or two features
influence the predictions of a TabPFN model. Built on scikit-learn’s
`PartialDependenceDisplay`, it supports both partial dependence (average effect)
and ICE (individual conditional expectation) curves, making it easy to interpret
feature impact on model outputs.

## TabPFN shapiq

``shapiq`` is a library for computing Shapley-based explanations like Shapley values or Shapley
interactions for machine learning models. The library is a redesigned and improved version of
the well-known SHAP library that provides a more efficient and scalable implementation of Shapley
values and Shapley interactions. In addition, ``shapiq`` offers native support for interpreting
TabPFN by utilizing a remove-and-recontextualize paradigm of model interpretation tailored towards
in-context models.

Three entry points live in ``tabpfn_extensions.interpretability.shapiq``, each
choosing a different answer to "what does the model predict when feature j is
missing?":

| Entry point                           | How "missing" is handled                                                                 | Cost on v3               |
|---------------------------------------|------------------------------------------------------------------------------------------|--------------------------|
| ``get_tabpfn_nan_explainer``          | Set feature to ``NaN`` and let TabPFN absorb it natively                                 | **Fastest** (KV cache)   |
| ``get_tabpfn_imputation_explainer``   | Sample the feature from a background distribution (``"marginal"`` / ``"conditional"`` / …) | Fast (KV cache)          |
| ``get_tabpfn_explainer``              | Refit the model on just the present features (remove-and-recontextualize)                | Slow (no cache reuse)    |

All three wrap the same ``shapiq`` explainer surface, so the ``.explain()`` /
``.explain_X()`` API and the resulting ``InteractionValues`` objects are
identical — only the Shapley *value function* differs.

### Speeding up with the TabPFN v3 KV cache

The NaN and imputation explainers call ``predict_proba`` once per coalition
(or per batch of coalitions) while the training data stays constant. TabPFN
v3's ``fit_mode="fit_with_cache"`` builds the per-estimator KV cache at
``.fit()`` time, and pinning those caches on-device lets every subsequent
``predict`` skip the CPU↔GPU transfer:

```python
from tabpfn import TabPFNClassifier
from tabpfn_extensions.interpretability import shapiq as tpe_shapiq

clf = TabPFNClassifier(fit_mode="fit_with_cache", device="auto")
clf.fit(X_train, y_train)
# Keep the per-estimator KV caches on the GPU across predicts.
clf.executor_.keep_cache_on_device = True

explainer = tpe_shapiq.get_tabpfn_nan_explainer(
    model=clf, data=X_train, index="SV", max_order=1, class_index=1,
)
iv = explainer.explain(x=X_test[0], budget=2 ** X_train.shape[1])
```

This does **not** help ``get_tabpfn_explainer`` (refit): each coalition has
different training features, so the cache built at ``.fit()`` is not reused.

### Citation

```bibtex
@inproceedings{muschalik2024shapiq,
  title     = {shapiq: Shapley Interactions for Machine Learning},
  author    = {Maximilian Muschalik and Hubert Baniecki and Fabian Fumagalli and
               Patrick Kolpaczki and Barbara Hammer and Eyke H\"{u}llermeier},
  booktitle = {Advances in Neural Information Processing Systems},
  pages     = {130324--130357},
  url       = {https://openreview.net/forum?id=knxGmi6SJi},
  volume    = {37},
  year      = {2024}
}
```

```bibtex
@InProceedings{rundel2024interpretableTabPFN,
  author    = {David Rundel and Julius Kobialka and Constantin von Crailsheim and
               Matthias Feurer and Thomas Nagler and David R{\"u}gamer},
  title     = {Interpretable Machine Learning for TabPFN},
  booktitle = {Explainable Artificial Intelligence},
  year      = {2024},
  pages     = {465--476},
  url       = {https://link.springer.com/chapter/10.1007/978-3-031-63797-1_23}
}
```

## Migrating from the removed ``shap`` adapter

The ``tabpfn_extensions.interpretability.shap`` module was removed because
its SHAP-via-all-NaN-background computation is numerically identical (at
``budget = 2**n_features``) to the new ``get_tabpfn_nan_explainer`` from the
shapiq adapter — both compute Shapley values of the same value function
``v(S) = predict_proba(x with features ∉ S set to NaN)``.

Old:

```python
from tabpfn_extensions.interpretability import shap
shap_values = shap.get_shap_values(clf, X_test, attribute_names=feature_names)
shap.plot_shap(shap_values)
```

New:

```python
from tabpfn_extensions.interpretability import shapiq as tpe_shapiq

explainer = tpe_shapiq.get_tabpfn_nan_explainer(
    model=clf, data=X_train, index="SV", max_order=1, class_index=1,
)
iv_per_row = explainer.explain_X(
    X_test, budget=2 ** X_train.shape[1]
)
iv_per_row[0].plot_waterfall(feature_names=feature_names)  # shapiq plots
```
