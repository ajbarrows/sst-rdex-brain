"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from .pipelines.data_management.pipeline import *
from .pipelines.rdex_prediction.pipeline import *
from .pipelines.phenotype_prediction.pipeline import *
from .pipelines.produce_plots.pipeline import *


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    # pipelines = find_pipelines()
    # pipelines["__default__"] = sum(pipelines.values())

    all_pipes = \
    data_management_pipe \
    + map_allregressor \
    + predict_phenotype_pipe \
    + crosspredict_pipe \
    + effectsize_plots \
    + rdex_prediction_featimp_plots_pipe \
    + make_supplement \
    

    

    return {
        "__default__": all_pipes,
        'data_management': data_management_pipe,
        'rdex_prediction': map_allregressor,
        'predict_phenotype': predict_phenotype_pipe,
        'crosspredict': crosspredict_pipe,
        'effectsize_plots': effectsize_plots,
        "brainmaps": rdex_prediction_featimp_plots_pipe,
        'supplement': make_supplement
    }
