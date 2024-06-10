from kedro.pipeline import Pipeline, node, pipeline

from .nodes import *

# def create_pipeline(**kwargs) -> Pipeline:
#     return pipeline([])


sst_behavior = pipeline(
    [
        node(
            load_behavioral,
            inputs='sst_behavior',
            outputs='sst_behavior_prepped'
        ),
        node(
            assemble_sst_behavioral,
            inputs=[
                'sst_behavior_prepped', 
                'params:beh_conditions',
                'params:beh_rt_vars' 
                ],
            outputs='sst_assembled'
        ),
        node(
            flag_uvm_sst_exclusion,
            inputs='sst_assembled',
            outputs='uvm_flagged'
        ),
        node(
            flag_sst_beh_exclusion,
            inputs=['uvm_flagged', 'sst_behavior_prepped'],
            outputs='behavior_flagged'
        ),
        node(
            flag_sst_mri_exclusion,
            inputs=['behavior_flagged', 'mri_qc'],
            outputs='sst_metrics'
        ),
        node(
            subset_baseline,
            inputs='sst_metrics',
            outputs='sst_metrics_bl'
        ),
        node(
            apply_exclusion_criteria,
            inputs='sst_metrics_bl',
            outputs='sst_metrics_filtered'
        )
    ]
)

assemble_phenotype_dataset = pipeline(
    [
        node(
            abcd5_loader,
            inputs=['mental_health', 'params:mh_keywords'],
            outputs='mh_loaded'
        ),
        node(
            abcd5_loader,
            inputs=['neurocognition', 'params:neurocog_keywords'],
            outputs='neurocog_loaded'
        ),
        node(
            merge_dataframes,
            inputs=['mh_loaded', 'neurocog_loaded'],
            outputs='survey_merged'
        ),
        node(
            assemble_phenotype_df,
            inputs=[
                'survey_merged',
                'params:bisbas_vars',
                'params:upps_vars',
                'params:nihtb_vars', 
                'params:cbcl_vars'
            ],
            outputs='phenotype_assembled'
        ),
        node(
            filter_phenotype,
            inputs=[
                'phenotype_assembled',
                'weigard_map',
                'params:timepoint'],
            outputs=['rdex', 'phenotype']
        )
    ]
)


data_management_pipe = sst_behavior + assemble_phenotype_dataset