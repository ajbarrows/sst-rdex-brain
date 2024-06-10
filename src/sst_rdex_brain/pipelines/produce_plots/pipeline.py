"""
This is a boilerplate pipeline 'produce_plots'
generated using Kedro 0.18.10
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import *
from .rdex_predict_plots import *
from .crosspredict_plots import *
from .helper_functions import *
from .phenotype_predict_plots import *



rdex_prediction_effectsize_plot_pipe = pipeline(
    [
        node(
            func=load_map_results,
            inputs=[
                'map_all_regressor_summary',
                "params:process_map",
                'params:target_map'
            ],
            outputs='rdex_effectsize_df'
        ),
        node(
            func=make_model_effectsize_plot,
            inputs='rdex_effectsize_df',
            outputs=None
        )
    ]
)

rdex_prediction_featimp_plots_pipe = pipeline(
    [
        node(
            filter_contribution_scores,
            inputs=[
                "map_model_contribution_scores",
                "params:target_map",
                "params:contribution_pval_threshold",
                "params:contribution_plot_metric"
            ],
            outputs=['cont_df_fdr', 'cont_df']
        ),
        node(
            separate_by_activation_direction,
            inputs=['cont_df', 'cont_df_fdr'],
            outputs=['pos_group', 'neg_group']
        ),
        node(
            generate_plotting_collections,
            inputs=[
                'pos_group', 
                'neg_group',
                'params:contribution_plot_metric'
                ],
            outputs=['pos_collection', 'neg_collection']
        ),
        # for supplement
        # node(
        #     generate_contribution_plots,
        #     inputs=[
        #         'cont_df_fdr',
        #         'pos_collection',
        #         'neg_collection',
        #         'params:posneg_mode_full'
        #     ],
        #     outputs=None
        # ),
        node(
            make_compare_plot,
            inputs=[
                'pos_collection',
                'params:compare_targets',
                'params:compare_conditions'
            ],
            outputs=None
        ),
        # node(
        #     load_reg_coef,
        #     inputs='map_model_contribution_scores',
        #     outputs='avg_reg_to_plot'
        # ),
        node(
            make_two_parameter_correlation_plot,
            inputs=['weigard_map', 'pos_group'],
            outputs=None
        )
    ]
)


rdex_prediction_plots_pipe = pipeline(
    [
        rdex_prediction_effectsize_plot_pipe,
        rdex_prediction_featimp_plots_pipe
    ]
)

crosspredict_plot_pipe = pipeline(
    [
        node(
            transform_crosspredict_features,
            inputs=[
                'crosspredict_results',
                'params:process_map',
                'params:target_map'
            ],
            outputs='crosspredict_plot_df'
        ),
        node(
            make_crosspredict_plot,
            inputs='crosspredict_plot_df',
            outputs=None
        )
    ]
)


phenotype_plots_pipe = pipeline(
    [
        node(
            assign_phenotype_labels,
            inputs=[
                'phenotype_full_summary',
                'phenotype_rdex_summary',
                'phenotype_empirical_summary'
            ],
            outputs='phenotype_effect'
        ),
        node(
            filter_spans_zero,
            inputs=[
                'phenotype_effect',
                'params:phenotype_effectsize_threshold'
            ],
            outputs='phenotype_effect_filtered'
        ),
        node(
            generate_limited_phenotype_effsize_plot,
            inputs=[
                'phenotype_effect_filtered',
                'params:phenotype_plotname_map_keyed',
                'params:nihtb_vars',
                'params:cbcl_vars'
            ],
            outputs=None
        )

    ]   

)


make_supplement = pipeline(
    [
        node(
            load_phenotype_summaries,
            inputs=[
                'phenotype_rdex_summary',
                'phenotype_empirical_summary',
                'phenotype_full_summary',
                'params:model_labels'
            ],
            outputs='phenotype_summaries'
        ),
        node(
            apply_category_labels,
            inputs= [
                'phenotype_summaries',
                'params:bisbas_vars',
                'params:upps_vars',
                'params:nihtb_vars',
                'params:cbcl_vars'
            ],
            outputs='phenotype_summaries_labeled'
        ),
        node(
            make_phenotype_supplement_table,
            inputs=[
                'phenotype_summaries_labeled',
                'params:phenotype_plotname_map'
            ],
            outputs='phenotype_supplement_table'
        ),
        node(
            load_reg_coef,
            inputs='map_model_contribution_scores',
            outputs='avg_reg'
        ),
        node(
            make_brainmap_correlation_plot,
            inputs=[
                'avg_reg',
                'params:condition_by_target_columns',
                'params:condition_by_target_index'
            ],
            outputs=None
        ),
        node(
            make_brainmap_correlation_plot,
            inputs=[
                'avg_reg',
                'params:target_by_condition_columns',
                'params:target_by_condition_index'
            ],
            outputs=None
        ),
        node(
            make_parameter_correlation_plot,
            inputs='weigard_map',
            outputs=None
        )

    ]
)


effectsize_plots = rdex_prediction_effectsize_plot_pipe + crosspredict_plot_pipe + phenotype_plots_pipe 