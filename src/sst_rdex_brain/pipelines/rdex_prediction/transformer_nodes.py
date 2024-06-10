import numpy as np
import pandas as pd
import BPt as bp

from scipy.stats import ttest_1samp, zscore
from statsmodels.stats.multitest import fdrcorrection


def merge_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    return df1.join(df2)

def get_scope(df: pd.DataFrame) -> list:
    return list(df.columns)

def assemble_scopes(predictors: pd.DataFrame, targets: pd.DataFrame):
    
    scopes = {}

    scopes['predictors'] = list(predictors)
    # scopes['covariates'] = list(conf)

    targets = list(targets)

    return scopes, targets

def merge_scopes(scopes, confound_scope):
    
    scopes['covariates'] = list(confound_scope)

    return scopes

def merge_predictors(df: pd.DataFrame, mri_data: pd.DataFrame):

    df = (df
          .reset_index()
          .set_index('subjectkey')
          .join(mri_data.reset_index().set_index('subjectkey'))
    )

    return df

def merge_target_dataset(df1: pd.DataFrame, df2: pd.DataFrame):
    '''Merge targets and limit to subjects who appear in both data sets.
    See sst_rdex.data_management for details.'''

    df = df1.join(df2, how='inner')
    targets = list(df.columns)

    # align NDA SSRT with RDEX (i.e., convert to seconds)
    MS_PER_SEC=1000
    df['issrt'] = df['issrt'] / MS_PER_SEC

    return [df, targets]

def merge_resid(df1: pd.DataFrame, df2: pd.DataFrame, targets: list, targets_resid: list):

    df = df1.join(df2)
    targets.extend(targets_resid)

    return df, targets

def merge_all(targets_df: pd.DataFrame, predictors_df: pd.DataFrame):
    return targets_df.join(predictors_df)

def get_result_summary(results: bp.CompareDict) -> pd.DataFrame:
    return results.summary().reset_index()

def get_targets(df: pd.DataFrame) -> list:
    return list(df)


def keep_only_contrasts(df):
    return df.filter(like='vs')


def get_feature_importance_contribution(results: bp.CompareDict) -> pd.DataFrame:
    
    df = pd.DataFrame()
    
    for l, m in results.items():

        lab = l.__dict__['options']

        if len(lab) == 1:
            scope = 'all'
            target = lab[0].__dict__['name']
        else:
            scope = lab[0].__dict__['name']
            target = lab[1].__dict__['name']

        if scope == 'covariates':
            continue
        else:
            coefs = m.get_fis()

            coefs.insert(0, 'target', target)

            df = pd.concat([df, coefs])
        
        # mean across folds
    df = df.groupby('target').agg('mean')

    return df

def t_test_contribution(contribution):

    t_df = contribution.agg(ttest_1samp, popmean=0, axis=0)
    t_df['metric']  = ['t_stat', 'p']

    return t_df

def compute_contribution(df, vertex):

    filter_str = 'mri|motion'
    df.drop(list(df.filter(regex=filter_str)), axis=1, inplace=True)

    df_contribution = pd.DataFrame()
    vertex = vertex.filter(df.columns)

    avg_activation = pd.DataFrame(vertex.mean()).T
    avg_activation.insert(0, 'metric', 'avg_activation')

    for r in df.iterrows():

        target = r[0]
        contribution = r[1] * vertex

        t_df = t_test_contribution(contribution)
        t_df.insert(0, 'target', target)

        df_contribution = pd.concat([df_contribution, t_df])

    df.insert(0, 'metric', 'avg_reg_coef')
    df = df.reset_index(names='target')

    df_contribution = pd.concat([df_contribution, df, avg_activation])

    return df_contribution

def reshape_contributions(df: pd.DataFrame) -> pd.DataFrame:

    filter_str = 'mri|motion'
    df.drop(list(df.filter(regex=filter_str)), axis=1, inplace=True)

    melted = pd.melt(df, id_vars=['target', 'metric'])

    if len(melted) > 1000:
        melted['variable'] = melted['variable'].str.replace('ssdstop', 'ssd_stop')
        tmp = melted['variable'].str.split('_', expand=True)

        tmp['condition'] = tmp[0] + '_' + tmp[1]
        tmp['brain'] = tmp[2] + '_' + tmp[3]

        tmp = tmp.iloc[:, -2:]

    else:
        tmp = melted['variable'].str.split('__', expand=True)
        tmp.columns = ['condition', 'brain']


    melted = melted.drop('variable', axis=1).assign(**tmp)

    melted = melted.pivot(
        index=['target', 'metric', 'condition'], 
        columns='brain', 
        values='value')
    
    return melted.reset_index()


