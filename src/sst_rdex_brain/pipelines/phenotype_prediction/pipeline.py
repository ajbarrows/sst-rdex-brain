"""
This is a boilerplate pipeline 'phenotype_prediction'
generated using Kedro 0.18.10
"""

from kedro.pipeline import Pipeline, node, pipeline

from sst_rdex_brain.pipelines.phenotype_prediction.nodes import *
from sst_rdex_brain.pipelines.phenotype_prediction.loader_nodes import *

from sst_rdex_brain.pipelines.rdex_prediction.nodes import fit_map_model, define_regression_pipeline

load_just_rdex_pipe = pipeline(
    [
        node(
            load_just_rdex,
            inputs='rdex_target_df',
            outputs='rdexonly_pred_df'
        )
    ]
)

load_just_empirical_pipe = pipeline(
    [
        node(
            load_just_empirical,
            inputs='rdex_target_df',
            outputs='empirical_pred_df'
        )
    ]
)

predict_phenotype_pipe = pipeline(
    [   
        node(
            load_confounds,
            inputs='confounders',
            outputs='conf'
        ),
        node(
            load_phenotype,
            inputs='phenotype',
            outputs='pheno'
        ),
        node(
            min_max_scale,
            inputs='pheno',
            outputs='pheno_scaled'
        ),
        node(
            get_scopes,
            inputs=['pred_df', 'pheno_scaled', 'conf'],
            outputs=['predictor_scopes', 'phenotype_targets']
        ),
        node(
            join_all,
            inputs=['pred_df', 'pheno_scaled', 'conf'],
            outputs='df'
        ),
        node(
            prepare_dataset,
            inputs=['df', 'phenotype_targets', 'predictor_scopes'],
            outputs='ds'
        ),
        node(
            fit_map_model,
            inputs=['ds', 'predictor_scopes'],
            outputs='phenotype_full_results'
        ),
        node(
            get_result_summary,
            inputs='phenotype_full_results',
            outputs='phenotype_full_summary'
        ),
        # node(
        #     run_permutation_tests,
        #     inputs='phenotype_full_results',
        #     outputs='phenotype_permutation'
        # ),
        node(
            get_feature_importance,
            inputs='phenotype_full_results',
            outputs='phenotype_feat_imp'
        )
    ]
)


reshape_phenotype_results = pipeline(
    [
        node(
            apply_category_labels,
            inputs=[
                'phenotype_full_summary',
                'params:bisbas_vars',
                'params:upps_vars',
                'params:nihtb_vars',
                'params:cbcl_vars'
            ],
            outputs='phenotype_summary_relabeled'
        ),
        node(
            join_permutation_pvals,
            inputs=[
                'phenotype_summary_relabeled',
                'phenotype_permutation'
            ],
            outputs='phenotype_summary_joined'
        ),
        node(
            apply_fdr_correction_groupwise,
            inputs='phenotype_summary_joined',
            outputs='phenotype_summary_fdr_corrected'
        )
    ]
)


predict_phenotype_rdex = pipeline(
    pipe = [
        load_just_rdex_pipe,
        predict_phenotype_pipe
    ],
    inputs={'pred_df': 'phenotype_rdex.rdexonly_pred_df'},
    outputs={
        'phenotype_full_results': 'phenotype_rdex_results',
        'phenotype_full_summary': 'phenotype_rdex_summary',
        'phenotype_feat_imp': 'phenotype_rdex_feat_imp'
    },
    namespace='phenotype_rdex'
)

predict_phenotype_empirical = pipeline(
    pipe = [
        load_just_empirical_pipe,
        predict_phenotype_pipe
    ],
    inputs={'pred_df': 'phenotype_empirical.empirical_pred_df'},
    outputs={
        'phenotype_full_results': 'phenotype_empirical_results',
        'phenotype_full_summary': 'phenotype_empirical_summary',
        'phenotype_feat_imp': 'phenotype_empirical_feat_imp'
    },
    namespace='phenotype_empirical'
)

predict_phenotype_both = pipeline(
    pipe = [
        predict_phenotype_pipe
    ],
    inputs = {'pred_df': 'phenotype_both.rdex_target_df'},
    namespace='phenotype_both'
)


predict_phenotype_pipe = predict_phenotype_rdex + predict_phenotype_empirical + predict_phenotype_both