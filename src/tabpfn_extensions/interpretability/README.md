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
in-context models. The ``shapiq`` library and the paper introducing the improved Shapley value
computation for TabPFN can be cited as follows:

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
  title     = {Interpretable Machine Learning for TabPFN},
  booktitle = {Explainable Artificial Intelligence},
  year      = {2024},
  pages     = {465--476},
  url       = {https://link.springer.com/chapter/10.1007/978-3-031-63797-1_23}
}

```

## TabPFN SHAP

``shap`` is a library for computing Shapley values explanations for machine learning models.
``shap`` constructs these explanations by imputing missing values with randomly drawn samples from
a background distribution. The ``shap`` library can be cited as follows:

```bibtext
@inproceedings{DBLP:conf/nips/LundbergL17,
  author       = {Scott M. Lundberg and
                  Su{-}In Lee},
  title        = {A Unified Approach to Interpreting Model Predictions},
  booktitle    = {Advances in Neural Information Processing Systems 30},
  pages        = {4765--4774},
  year         = {2017},
  url          = {https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html},
}
```
