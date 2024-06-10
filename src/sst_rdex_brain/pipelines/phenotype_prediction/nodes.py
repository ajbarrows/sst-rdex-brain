import pandas as pd
import numpy as np

import BPt as bp

from scipy.stats import ttest_1samp, zscore
from statsmodels.stats.multitest import fdrcorrection

def get_scopes(predictors: pd.DataFrame, pheno: pd.DataFrame, 
               conf: pd.DataFrame):
    
    scopes = {}

    scopes['predictors'] = list(predictors)
    scopes['covariates'] = list(conf)

    targets = list(pheno)

    return scopes, targets

def min_max_scale(df):
    
    df = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    
    return df

def join_all(predictors: pd.DataFrame, pheno: pd.DataFrame, 
               conf: pd.DataFrame) -> pd.DataFrame:
    
    df = (pheno
          .join(conf)
          .join(predictors))
    
    return df

def prepare_dataset(df: pd.DataFrame, targets: list, scopes: dict) -> bp.Dataset:

    df = df[~df.index.duplicated(keep='first')]

    ds = bp.Dataset(df, targets=targets)
    for k, v in scopes.items():
         print(k, v)
         ds.add_scope(v, k, inplace=True)

    ds = ds.auto_detect_categorical()
    ds = ds.ordinalize('category')

    ds.dropna(inplace=True)

    ds = ds.set_test_split(.2, random_state=42)

    return ds

def get_result_summary(results: bp.CompareDict) -> pd.DataFrame:
    
    return results.summary().reset_index()

def run_permutation_tests(results: bp.CompareDict) -> list:

    p_values = {}
    p_scores = {}

    for k, v in results.items():

        lab = k.__dict__['options']

        target = lab[0].__dict__['name']

        
        pvals, pscores = v.run_permutation_test(n_perm=100)

        p_values[target] = pvals
        p_scores[target] = pscores

    
    out = {'p_values': p_values, 'p_scores': p_scores}

    return out


def t_test(x): 
    res = ttest_1samp(x, popmean=0, axis=0, keepdims=True)
    p = res.pvalue
    t = res.statistic

    return t, p

def get_averages(df_reshaped: pd.DataFrame) -> pd.DataFrame:

    def cohen_d(x) : return np.mean(x, axis=0) / np.std(x, axis=0)

    mean = df_reshaped.agg('mean')
    std = df_reshaped.agg('std')
    cohen = df_reshaped.apply(cohen_d)

    mean['metric'] = 'mean'
    std['metric'] = 'std'
    cohen['metric'] = 'cohen'

    df = pd.concat([mean, std, cohen], axis=0)

    return df

def get_significance(df):


    t, p = t_test(df)

    z_score = zscore(df, axis=1)
    z_avg = z_score.agg('mean')

    t_df = pd.DataFrame(t)
    p_df = pd.DataFrame(p)

    # n_tests = p_df.shape[1]
    # p_bonferroni = p_df * n_tests
    
    # p_fdr = fdrcorrection(np.squeeze(p_df))[1]
    # p_fdr_df = pd.DataFrame(columns=df.columns)
    # p_fdr_df.loc[0] = p_fdr

    z_df = pd.DataFrame(columns=df.columns)
    z_df.loc[0] = z_avg

    t_df.columns = p_df.columns = df.columns

    t_df.insert(0, 'metric', 't_stat')
    p_df.insert(0, 'metric', 'p')
    # p_bonferroni.insert(0, 'metric', 'p_bonf')
    # p_fdr_df.insert(0, 'metric', 'p_fdr')
    z_df.insert(0, 'metric', 'z')

    out = pd.concat([t_df, p_df, z_df])


    return out


def get_feature_importance(results: bp.CompareDict) -> pd.DataFrame:
    
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
            
            avg = coefs.agg(['mean', 'std']).reset_index(names='metric')
            sig = get_significance(coefs)
            
            sig.insert(0, 'target', target)
            avg.insert(0, 'target', target)
        
            df = pd.concat([df, avg, sig])

    return df


def apply_category_labels(res: pd.DataFrame, 
                          bisbas_vars: dict, upps_vars: dict, 
                          nihtb_vars: dict, cbcl_vars: dict):
    cat_labels = {
        bisbas_vars.values(): 'Behavioral Inhibition/Approach System (BIS/BAS)',
        upps_vars.values(): 'UPPS-P Impulsive Behavior Scale',
        nihtb_vars.values(): 'NIH ToolBox (Uncorrected)',
        cbcl_vars.values(): 'CBCL Scores (Raw)'
    }

    res['category'] = res['target']

    for k, v in cat_labels.items():

        res['category'] = np.where(res['category'].isin(k), v, res['category'])

    return res


def join_permutation_pvals(res: pd.DataFrame, perm: pd.DataFrame) -> pd.DataFrame:

    p_vals = perm['p_values']
    perm_df = pd.DataFrame()

    for k, v in p_vals.items():
        tmp = pd.DataFrame(v, index=[k])
        tmp = (tmp
            .drop(columns='neg_mean_squared_error')
            .rename(columns={'r2': 'p'})
        )
        
        perm_df = pd.concat([perm_df, tmp])

    res = res.set_index('target').join(perm_df).reset_index()

    return res

def apply_fdr_correction_groupwise(res: pd.DataFrame):

    # apply FDR correction by group

    tmp = res[['target', 'category', 'p']]
    categories = pd.unique(tmp['category'])

    grp = tmp.groupby('category')

    df_out = pd.DataFrame()
    for c in categories:
        
        struct = grp.get_group(c)
        fdr = fdrcorrection(struct['p'].squeeze())[1]
        fdr_df = pd.DataFrame({'target': struct['target'], 'p_fdr': fdr})

        df_out = pd.concat([df_out, fdr_df])

    res = res.set_index('target').join(df_out.set_index('target')).reset_index()

    return res