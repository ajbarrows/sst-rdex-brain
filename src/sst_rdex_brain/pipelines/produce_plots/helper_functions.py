import pandas as pd
import numpy as np

from statsmodels.stats.multitest import fdrcorrection

def format_p_value(p):
    if p < 0.05:
        fmt = "*"
    elif p < 0.01:
        # fmt = "p<.001"
        fmt = "**"
    elif p< .001:
        fmt = "***"
    elif p> 0.05:
        fmt = "ns"

    return fmt

def _fdr_pval_only(x):
    res, p_fdr = fdrcorrection(x)
    
    return p_fdr

def mask_non_significant(df: pd.DataFrame, method='fdr', threshold=0.05, return_metric='mean'):

    tmp = df[df['metric'] == return_metric]
    p_val = df[df['metric'] == 'p']

    idx = ['target', 'metric', 'condition']
    p_fdr = (p_val
            .set_index(idx)
            .apply(_fdr_pval_only, axis=1, result_type='broadcast')
            .reset_index()
            .drop(columns='metric')
            .set_index(['target', 'condition'])
    )

    out = (tmp
        .drop(columns='metric')
        .set_index(['target', 'condition'])
        .mask(p_fdr>threshold, 0))
    
    out.insert(0, 'metric', return_metric)

    return out.reset_index()


def get_fullrang_minmax(df, target):

    sub = df[df['target'] == target]
    sub = sub.set_index(['target', 'condition', 'metric'])

    mi = sub.min().min()
    ma = sub.max().max()

    abs_max = max(np.abs(mi), np.abs(ma))

    return -abs_max, abs_max

def rename_target(df, target_map):
    return df.replace(target_map)

