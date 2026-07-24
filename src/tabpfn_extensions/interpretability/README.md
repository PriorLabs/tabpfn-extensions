# TabPFN Interpretability

## TabPFN Decoder-Head Readout

TabPFN classifies with an attention-based retrieval head (`ManyClassDecoder`):
each test row attends to the training rows and predicts the attention-weighted
average of their labels. `get_decoder_readout` recovers those per-training-row
attention weights, so a prediction can be read as a label-vote over training
points — `P(class c)` is the sum of a row's weights over training rows of class
`c`.

```python
from tabpfn_extensions import TabPFNClassifier
from tabpfn_extensions.interpretability import get_decoder_readout, class_vote

clf = TabPFNClassifier().fit(X_train, y_train)

# weights: (n_test, n_train), each row sums to 1 (averaged over heads and the
# ensemble). train_indices maps the columns back to the fitted training rows.
weights, train_indices = get_decoder_readout(clf, X_test)

# Collapse by training label; averaged over the ensemble this reproduces
# predict_proba up to the head's log-clamping, at the default
# softmax_temperature=0.9 and balance_probabilities=False (both are applied
# downstream of this readout, so non-default values widen the gap).
votes, classes = class_vote(weights, y_train)
```

Only the local `tabpfn` backend is supported (the client/API backend does not
expose the model internals), and row subsampling
(`TabPFNClassifier(..., inference_config={"SUBSAMPLE_SAMPLES": ...})`) is not
supported since the weight columns would no longer align to a single set of
training rows; `get_decoder_readout` raises `NotImplementedError` if it detects
subsampling is active.

See `examples/interpretability/decoder_readout_example.py`, which projects the
model's training embeddings to 2D and draws, for queries spanning the confidence
range, the lines from each query to its most-attended training rows (colored by
class, scaled by vote weight).

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

We expose three adapters:

* ``get_tabpfn_explainer`` — *remove-and-recontextualize* (Rundel et al. 2024).
  TabPFN is re-fit for every coalition, so the KV cache cannot be reused
  across coalitions — expect this path to be substantially slower than the
  imputation-based one below.

* ``get_tabpfn_imputation_explainer`` — *imputation-based* removal (marginal / conditional /
  baseline). The training set is fixed across coalitions, so the KV-cache fast path
  applies. Construct your TabPFN model with ``fit_mode="fit_with_cache"`` (set BEFORE
  ``.fit()``):

  ```python
  clf = TabPFNClassifier(fit_mode="fit_with_cache")
  clf.fit(X_train, y_train)
  explainer = get_tabpfn_imputation_explainer(model=clf, data=X_train)
  ```

  The wrapper warns at construction time if the model isn't configured this way.

* ``get_tabpfn_inf_explainer`` — *inf-masking* removal. A masked feature is set to
  ``+inf`` and TabPFN's native missing-value handling absorbs it as "missing" — no
  sampling, one forward pass per coalition. Like the imputation path the training set
  is fixed, so the KV cache applies. It requires the model to be built with
  ``inference_config={"PASSTHROUGH_INF": True}`` (``tabpfn>=8.1.0``) so ``+inf`` reaches
  the model instead of being rejected at validation; unlike ``NaN``, which TabPFN's
  preprocessing transforms before it reaches the model, ``+inf`` is carried through:

  ```python
  clf = TabPFNClassifier(
      fit_mode="fit_with_cache",
      inference_config={"PASSTHROUGH_INF": True},
  )
  clf.fit(X_train, y_train)
  explainer = get_tabpfn_inf_explainer(model=clf, data=X_train)
  ```

  The wrapper raises at construction time if ``PASSTHROUGH_INF`` isn't enabled.

For SHAP-style plots (waterfall, beeswarm, summary, dependence) you have two options:

1. **Use shapiq's own visualizations** on the returned ``InteractionValues`` object:
   ``iv.plot_force()``, ``iv.plot_waterfall()``, ``iv.plot_network()``,
   ``iv.plot_si_graph()``, etc.

2. **Use the SHAP library's plotting** via the bridge helper. Run shapiq's
   ``.explain()`` over a batch of rows and wrap the result in a
   ``shap.Explanation`` in one call:

   ```python
   from tabpfn_extensions.interpretability import shapiq_to_shap_explanation

   explanation = shapiq_to_shap_explanation(
       explainer, X_explain, budget=256, feature_names=feature_names,
   )
   shap.plots.waterfall(explanation[0])
   ```

   ``shapiq_to_shap_explanation`` extracts first-order Shapley values from
   shapiq's output and wraps them in a ``shap.Explanation``. Requires
   ``pip install shap`` — kept out of the ``interpretability`` extra by
   design (shapiq is the runtime dependency; shap is opt-in for plotting).

See ``examples/interpretability/shapiq_example.py`` and ``shap_example.py``
for both paths.

The ``shapiq`` library and the paper introducing the improved Shapley value computation
for TabPFN can be cited as follows:

```bibtext
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
and
```bibtext
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

The original ``shap`` library — still used here for plotting via ``shap.Explanation`` —
can be cited as:

```bibtext
@inproceedings{DBLP:conf/nips/LundbergL17,
  author       = {Scott M. Lundberg and Su{-}In Lee},
  title        = {A Unified Approach to Interpreting Model Predictions},
  booktitle    = {Advances in Neural Information Processing Systems 30},
  pages        = {4765--4774},
  year         = {2017},
  url          = {https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html},
}
```
