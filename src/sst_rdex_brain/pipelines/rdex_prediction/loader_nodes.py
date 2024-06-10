import numpy as np
import pandas as pd

from typing import Any, Callable, Dict, List

def load_weigard(df: pd.DataFrame) -> pd.DataFrame:
    '''Load RDEX-ABCD parameter estimates (e.g., maximum a posteriori).'''

    if df.index.name != 'subjectkey':
        df.index.name = 'subjectkey'

    df = df.drop(columns=['tf.natural', 'gf.natural'])

    return df

def subset_weigard(df: pd.DataFrame) -> pd.DataFrame:
    '''For crosspredict pipe. Experimental.'''

    # cols_to_drop = ['EEA', 'SSRT']
    cols_to_drop = ['vT', 'vF', 'SSRT']
    
    return df.drop(columns=cols_to_drop)

def load_sst_metrics(df: pd.DataFrame, sst_metrics: list) -> pd.DataFrame:
    '''Load SST "go" metrics from NDA performance data.'''

    if df.index.name != 'subjectkey':
        df.index.name = 'subjectkey'

    return df[sst_metrics]


def load_scanner(df):
    '''Load baseline scanner serial number.'''
    
    df = df[df['eventname'] == 'baseline_year_1_arm_1']
    df = df[['subjectkey', 'mri_info_deviceserialnumber']]

    return df.set_index('subjectkey')

def load_motion(df):
    '''Load baseline average head motion.'''

    df = df[df['eventname'] == 'baseline_year_1_arm_1']
    df = df[['subjectkey', 'iqc_sst_all_mean_motion']]

    return df.set_index('subjectkey')

def load_inhouse_fmri_predictors_nowall(partitioned_input: Dict[str, Callable[[], Any]]):
    '''Load NERVE Lab-processed ROI contrasts.'''

    df = None

    for partition_key, partition_load_func in sorted(partitioned_input.items()):
        partition_data = partition_load_func()  # load the actual partition data
        partition_data = partition_data.set_index('subjectkey').iloc[:, 1:]
        partition_key = partition_key.replace('.csv', '')
        partition_data.columns = [f'{partition_key}__{x}' for x in partition_data.columns]
        partition_data = partition_data[[x for x in partition_data.columns if '_wall' not in x]]
        # concat with existing result
        if df is None:
            df = partition_data
        else:
            df = df.join(partition_data)

    unique_contrasts = [x.replace('.csv', '') for x in partitioned_input.keys()]

    scopes = {}

    for u in unique_contrasts:

        scopes[u] = []

        for c in df.columns:

            if u in c:

                scopes[u].append(c)

    return [df, scopes]

def load_sst_regressors(partitioned_input: Dict[str, Callable[[], Any]], 
                        pguids: pd.DataFrame) -> pd.DataFrame:
    '''Load NERVE Lab-processed vertexwise regressors.'''

    df = pd.DataFrame()

    rep_string = '_cleaned_norm_base'
    MISS_PROPORTION = 0.75

    for partition_key, partition_load_func in sorted(partitioned_input.items()):
        p_df = partition_load_func()  # load the actual partition data
        p_df.columns = [c.replace(rep_string, '') for c in p_df.columns]

        df = pd.concat([df, p_df], axis=1)


    subjects = ["NDAR_" + id for id in list(pguids)]
    df.insert(0, 'subjectkey', subjects)
    df = df.set_index('subjectkey')

    # drop columns with less than (1 - MISS_PROPORTION) values
    miss_thresh = MISS_PROPORTION * len(df)
    df.dropna(axis=1, thresh=miss_thresh, inplace=True)

    # then drop rows with any missing data
    df.dropna(inplace=True)

    unique_regressors = set([c.rsplit('_', 2)[0] for c in df.columns])
    
    scopes = {}

    for u in unique_regressors:

        scopes[u] = []

        for c in df.columns:

            if u in c:

                scopes[u].append(c)

    return [df, scopes]

def load_sst_regressor_contrasts(df: pd.DataFrame):
        
        unique_regressors = set([c.rsplit('_', 2)[0] for c in df.columns])
    
        scopes = {}

        for u in unique_regressors:

            scopes[u] = []

            for c in df.columns:

                if u in c:

                    scopes[u].append(c)

        return [df, scopes]
