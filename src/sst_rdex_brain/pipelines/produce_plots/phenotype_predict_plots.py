import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from .helper_functions import *

def assign_phenotype_labels(both: pd.DataFrame, rdex_only: pd.DataFrame, empirical_only: pd.DataFrame):

    both['group'] = 'RDEX + Empirical'
    rdex_only['group'] = 'RDEX Only'
    empirical_only['group'] = 'Empirical Only'

    return pd.concat([both, rdex_only, empirical_only])

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



def reshape_feature_importance(df, metric):

    def _drop_phenotype_covariates(df):

        filter_str = 'site|female|age'
        df.drop(list(df.filter(regex=filter_str)),
                      axis=1, inplace=True)

        return df
    
    df = _drop_phenotype_covariates(df)
    
    df = df[df['metric'] == metric]
    df = df.drop(columns='metric')
    df = df.set_index('target')
    df.columns = df.columns.to_flat_index()

    return df
    

def phenotype_fis_fdr_correction(fis_cat):

    tmp = reshape_feature_importance(fis_cat, 'p').reset_index()
    categories = pd.unique(tmp['category'])
    grp = tmp.groupby('category')
    p_fdr = pd.DataFrame()

    for c in categories:
        struct = grp.get_group(c)
        struct = struct.set_index(['target', 'category'])

        for name, values in struct.items():
            fdr = fdrcorrection(values.squeeze())[1]
            struct[name] = values

        p_fdr = pd.concat([p_fdr, struct])    

    p_fdr = p_fdr.reset_index().set_index(['target', 'category'])
    
    return p_fdr

def mask_feature_importance(df, phenotype_plotname_map, thresh=0.05):
    p_fdr = phenotype_fis_fdr_correction(df)
    mean = reshape_feature_importance(df, 'mean')

    mean = mean.reset_index().set_index(['target', 'category'])
    masked = mean.mask(p_fdr > 0.05, np.NaN)
    masked = masked.reset_index().replace(phenotype_plotname_map)

    return masked

def filter_spans_zero(df, filter_effect_size):

    tmp = df.copy()

    tmp['avg_sign'] = np.where(tmp['mean_scores_r2'] > 0, 'pos', 'neg')

    tmp['spans_zero'] = np.where(
        tmp['avg_sign'] == 'pos',
        (tmp['mean_scores_r2'] - tmp['std_scores_r2']) < 0,
        (tmp['mean_scores_r2'] + tmp['std_scores_r2']) > 0
    )

    df = df[~tmp['spans_zero']]

    if filter_effect_size:

        df = df[df['mean_scores_r2'] >= filter_effect_size]

            
    return df.reset_index(drop=True)


def mask_feature_importance(df, phenotype_plotname_map, thresh=0.05):
    p_fdr = phenotype_fis_fdr_correction(df)
    mean = reshape_feature_importance(df, 'mean')

    mean = mean.reset_index().set_index(['target', 'category'])
    mean = mean.reset_index().replace(phenotype_plotname_map)

    # return masked
    return mean

def process_fis(fis, bisbas_vars, upps_vars, nihtb_vars, cbcl_vars, phenotype_plotname_map, group):

    fis_cat = apply_category_labels(fis, bisbas_vars, upps_vars, nihtb_vars, cbcl_vars)
    masked = mask_feature_importance(fis_cat, phenotype_plotname_map)

    masked['group'] = group

    return masked


def apply_fis_processing(fis_full, fis_rdex, fis_empirical,bisbas_vars, upps_vars, nihtb_vars, cbcl_vars, phenotype_plotname_map_keyed):

    dfs = {
        'RDEX + Empirical': fis_full,
        'RDEX Only': fis_rdex,
        'Empirical Only': fis_empirical
    }

    fis = pd.DataFrame()

    for k, v in dfs.items():
        tmp = process_fis(
            v,
            bisbas_vars,
            upps_vars,
            nihtb_vars,
            cbcl_vars, 
            phenotype_plotname_map_keyed,
            k
        )

        fis = pd.concat([fis, tmp])

    return fis

def make_avg_feat_imp(df):

    df = (df
          .groupby('group')
          .mean()
          .reset_index()
          .melt(id_vars=['group']))
    
    return df


def make_feat_imp_radar_plot(df, ax, hue_order, title='Avg. Feature Importance for Phenotype Prediction'):

    variables = pd.unique(df['variable'])
    N = len(variables)

    categories = ['RDEX Only', 'Empirical Only',  'RDEX + Empirical']
    colors = ['#1f77b4','#aec7e8','#ff7f0e']


    radians = 2 * np.pi
    angles = [n / float(N) * radians for n in range(N)]
    angles += angles[:1]

    # instantiate plot
    ax.set_xticks(angles[:-1], variables)
    ax.set_rlabel_position(10)

    # plot circle to show 0
    rads = np.arange(0, (2 * np.pi), 0.01)
    zeros = np.zeros(len(rads))
    ax.plot(rads, zeros, 'k', alpha=.8)

    # turn grid off
    ax.patch.set_visible(False)
    ax.grid("off")
    ax.yaxis.grid(False)
    ax.spines['polar'].set_visible(False)

    for category, color in zip(categories, colors):

        tmp = df[df['group'] == category]
        values = tmp['value'].reset_index(drop=True).values
        values = np.append(values, values[0])

        ax.plot(angles, values, color=color)

    legend_labels = np.insert(categories, 0, 'Reference')
    ax.legend(legend_labels, bbox_to_anchor=(.01, 1.05))

def make_effectsize_plot(df, cat_labels, hue_order, phenotype_plotname_map_keyed, ax):
       # filter effectsize dataframe
    eff_size_plot = pd.DataFrame()
    for i,d in enumerate(cat_labels.items()):

        tmp = df[df['target'].isin(d[0])]
        tmp = tmp.replace(phenotype_plotname_map_keyed)

        eff_size_plot = pd.concat([eff_size_plot, tmp])

    eff_size_plot = eff_size_plot.sort_values('mean_scores_r2').reset_index()

    
    sns.barplot(
        eff_size_plot,
        x='target',
        y='mean_scores_r2',
        hue='group',
        hue_order=hue_order,
        palette='tab20',
        ax=ax)
    

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_ylim(-0.02, 0.16)
    ax.set_ylabel('$R^2$')
    ax.set_xlabel('')
    ax.get_legend().set_title(None)
    ax.legend(loc='upper left')

    ax.grid(linestyle=':')


    # effect size error bars
    x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
    y_coords = [p.get_height() for p in ax.patches]
    ax.errorbar(x=x_coords, y=y_coords, yerr=eff_size_plot["std_scores_r2"], fmt="none", c="k")

    return ax, eff_size_plot


def generate_limited_phenotype_effsize_plot(effect_size: pd.DataFrame, phenotype_plotname_map_keyed: dict,
                                            nihtb_vars: dict, cbcl_vars: dict,
                                            file='data/08_reporting/phenotype_effect_limited.png'):

    cat_labels = {
        # bisbas_vars.values(): 'Behavioral Inhibition/Approach System (BIS/BAS)',
        # upps_vars.values(): 'UPPS-P Impulsive Behavior Scale',
        nihtb_vars.values(): 'NIH ToolBox (Uncorrected)',
        cbcl_vars.values(): 'CBCL Scores (Raw)'
    }
    hue_order = ['RDEX Only', 'Empirical Only',  'RDEX + Empirical']
    sns.set_context('paper')

    fig, ax = plt.subplots()

    ax, effect_size = make_effectsize_plot(effect_size, 
                                           cat_labels, 
                                           hue_order, 
                                           phenotype_plotname_map_keyed, 
                                           ax)
    ax.set_title('Phenotype Prediction Effectsize Estimates')
    plt.savefig(file, dpi=300, bbox_inches='tight')


def load_phenotype_summaries(rdex, empirical, full, labels:list):

    out = pd.DataFrame()
    dfs = [rdex, empirical, full]

    for df, label in zip(dfs, labels):
        df['model'] = label
        out = pd.concat([out, df])

    return out

def make_phenotype_supplement_table(df: pd.DataFrame, phenotype_plotname_map: dict):

    df = (df
        .replace(phenotype_plotname_map)
        .sort_values(['model', 'category', 'mean_scores_r2'])
        .pivot(index=['category', 'target'], columns='model', values=['mean_scores_r2', 'std_scores_r2'])
    )

    return df.round(3)