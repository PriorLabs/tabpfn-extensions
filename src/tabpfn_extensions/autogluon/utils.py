

from tabpfn_extensions.autogluon.configs import TabPFNConfig
from tabpfn import TabPFNClassifier, TabPFNRegressor


def get_tabpfn(config: TabPFNConfig, **kwargs):
    """Get a combined TabPFN Model, e.g. stacking, bagging, ensemble, rf_pfn."""
    return tabpfn_model_type_getters[config.model_type](config, **kwargs)

def get_single_tabpfn(config: TabPFNConfig, **kwargs):
    """Get a single TabPFN model given the task type."""
    return tabpfn_task_type_models[config.task_type](**config.to_kwargs(), **kwargs)


tabpfn_model_type_getters = {
    "single": get_single_tabpfn,
    "single_fast": get_single_tabpfn,
}


tabpfn_task_type_models = {
    "multiclass": TabPFNClassifier,
    "regression": TabPFNRegressor,
}