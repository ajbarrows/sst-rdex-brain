"""
This is a boilerplate pipeline 'rdex_prediction'
generated using Kedro 0.18.10
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import *

from sst_rdex_brain.pipelines.rdex_prediction.pipeline import *
from sst_rdex_brain.pipelines.rdex_prediction.loader_nodes import *
from sst_rdex_brain.pipelines.rdex_prediction.transformer_nodes import *

load_confounds_pipe = pipeline(
    [
        node(
            load_scanner,
            inputs='scanner_info',
            outputs='scanner_prepped'
        ),
        node(
            load_motion,
            inputs='sst_head_motion',
            outputs='motion_prepped'
        ),
        node(
            merge_dataframes,
            inputs=['scanner_prepped', 'motion_prepped'],
            outputs='all_confounds'
        ),
        node(
            get_scope,
            inputs='all_confounds',
            outputs='confound_scope'
        )
    ]
)


load_rdex_targets_pipe = pipeline(
    [
        node(
            load_weigard,
            inputs='weigard_map',
            outputs='weigard_prepped'
        ),
        node(
            load_sst_metrics,
            inputs=[
                'sst_metrics_filtered',
                'params:sst_metrics'
            ],
            outputs='sst_metrics_prepped'
        ),
        node(
            merge_target_dataset,
            inputs=['weigard_prepped', 'sst_metrics_prepped'],
            outputs=['rdex_target_df', 'targets']
        )
    ]
)


load_regressors_pipe = pipeline(
    [
        node(
            load_sst_regressors,
            inputs=['sst_regressors', 'regressors_pguids'],
            outputs=['sst_prepped', 'mri_scopes']
        )
    ]
)

prepare_dataset_pipe = pipeline(
    [
        node(
            merge_scopes,
            inputs=['mri_scopes', 'confound_scope'],
            outputs='scopes'
        ),
        node(
            merge_predictors,
            inputs=['all_confounds', 'sst_prepped'],
            outputs='all_predictors'
        ),
        node(
            merge_all,
            inputs=['rdex_target_df', 'all_predictors'],
            outputs='full_df'
        ),
        node(
            prepare_dataset,
            inputs=['full_df', 'targets', 'scopes'],
            outputs='map_dataset'
        )

    ]
)



fit_model_pipe = pipeline(
    [   node(
            fit_map_model,
            inputs=[
                'map_dataset', 
                'scopes', 
                'params:model',
                'params:complete'],
            outputs='model_results'
        ),
        node(
            get_result_summary,
            inputs='model_results',
            outputs='model_summary'
        )

    ]
)


run_permutation_test_pipe = pipeline(
    [
        node(
            run_permutation_tests,
            inputs='map_all_regressor_results',
            outputs='map_permutation'
        )
    ]
)

# modular 
compute_contribution_pipe = pipeline(
    [
        node(
            get_feature_importance_contribution,
            inputs='model_results',
            outputs='fis'
        ),
        node(
            compute_contribution,
            inputs=['fis', 'sst_prepped'],
            outputs='contributions'
        ),

        node(
            reshape_contributions,
            inputs='contributions',
            outputs='contribution_scores'
        )
    ]
)




# All vertexwise regressors (ridge)
map_allregressor = pipeline(
    pipe = [
        load_confounds_pipe,
        load_rdex_targets_pipe,
        load_regressors_pipe,
        prepare_dataset_pipe,
        fit_model_pipe,
        compute_contribution_pipe
    ],
    parameters = {
        "params:model": "params:ridge",
        "params:complete": "params:complete"},
    outputs = {
        "model_results": "map_all_regressor_results",
        "model_summary": "map_all_regressor_summary",
        "contribution_scores": "map_model_contribution_scores"}

)



crosspredict_pipe = pipeline(
    [
        node(
            load_sst_metrics,
            inputs=[
                'sst_metrics_filtered', 
                'params:sst_metrics_predict'],
            outputs='cp_sst_metrics_prepped'
        ),
        node(
            get_targets,
            inputs='cp_sst_metrics_prepped',
            outputs='crosspredict_targets'
        ),
        node(
            load_weigard,
            inputs='weigard_map',
            outputs='weigard_full'
        ),
        node(
            subset_weigard,
            inputs='weigard_full',
            outputs='cp_weigard_prepped'
        ),
        node(
            merge_all,
            inputs=['sst_metrics_prepped', 'cp_weigard_prepped'],
            outputs='crosspredict_df'
        ),
        node(
            assemble_scopes,
            inputs=['cp_weigard_prepped', 'sst_metrics_prepped'],
            outputs='crosspredict_scopes'
        ),
        node(
            prepare_dataset_crosspredict,
            inputs=['crosspredict_df', 'crosspredict_targets'],
            outputs='crosspredict_ds'
        ),
        node(
            fit_crosspredict_model,
            inputs=['crosspredict_ds', 'crosspredict_scopes'],
            outputs='crosspredict_results'
        ),
        node(
            get_result_summary,
            inputs='crosspredict_results',
            outputs='crosspredict_results_summary'
        ),
        node(
            print_avg_effect,
            inputs='crosspredict_results_summary',
            outputs=None
        )
    
    ]
)



