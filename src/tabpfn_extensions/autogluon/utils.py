

from configs import TabPFNConfig


def get_tabpfn(config: TabPFNConfig, **kwargs):
    """Get a combined TabPFN Model, e.g. stacking, bagging, ensemble, rf_pfn."""
    return tabpfn_model_type_getters[config.model_type](config, **kwargs)


tabpfn_model_type_getters = {
    "single": get_single_tabpfn,
    "single_fast": get_single_tabpfn,
}