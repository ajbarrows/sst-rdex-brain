import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import neurotools.plotting as ntp
import seaborn as sns

from itertools import repeat
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.colorbar import make_axes
from scipy.stats import pearsonr

from .helper_functions import *



def load_map_results(df: pd.DataFrame, process_map: dict, target_map: dict) -> pd.DataFrame:

    df = df[~df['target'].isin(['tf.natural', 'gf.natural'])]

    df['process'] = df['target']
    df['process'] = df['process'].replace(process_map)
    df['target'] = df['target'].replace(target_map)

    # print(df)
    return df.sort_values(['process', 'mean_scores_r2'], ascending=False)

def make_model_effectsize_plot(df: pd.DataFrame, save=True):

    go_target = df[df['process'] == 'Go Process']['target']
    stop_target = df[df['process'] == 'Stop Process']['target']
    ssrt = df[df['process'] == 'SSRT']['target']
    go_empirical = ['Go choice accuracy', 'Correct-Go response time (RT)', 'Correct-Go RT variability']
    ssrt_empirical = ['Empirical SSRT']

    sns.set_context('talk')
    fig, ax = plt.subplots(1, 4, figsize=(24,9), dpi=100, sharey=True,
        gridspec_kw={'width_ratios':[8,3,1.25,.25]}
    )

    fig.suptitle("Cross-Validated Ridge Regression Model Performance (Parameter Predictability) (n=6460)")
    #Go Process
    go_color = '#77DD77'

    x_loc = 0
    starting_points = []

    for target in go_target:

        if target in go_empirical:
            hatch='/'
        else:
            hatch=None

        starting_points.append(x_loc)

        row = df[(df['target'] == target)]
        
        rects = ax[0].bar(x_loc, height=row['mean_scores_r2'], 
                        yerr=row['std_scores_r2'], width=.5, 
                        color=go_color, hatch=hatch)

        x_loc += 1


    ax[0].grid(linestyle=':')
    ax[0].set_xticks(np.array(starting_points))
    ax[0].set_xticklabels(go_target, rotation=45, ha='right')
    ax[0].set_title('Go Process')
    ax[0].set_ylabel('Model $R^2$')

    ax[0].spines[['top', 'right']].set_visible(False)

    # Stop process
    stop_color = "lightcoral"

    x_loc = 0
    starting_points = []
    for target in stop_target:

        starting_points.append(x_loc)

        row = df[(df['target'] == target)]
        rects = ax[1].bar(x_loc, height=row['mean_scores_r2'], yerr=row['std_scores_r2'], width=.5, color=stop_color)

        x_loc += 1

    ax[1].grid(linestyle=':')
    ax[1].set_xticks(np.array(starting_points))
    ax[1].set_xticklabels(stop_target, ha='right', rotation=45)
    ax[1].set_title('Stop Process')

    ax[1].spines[['top', 'left', 'right']].set_visible(False)

    # SSRT
    ssrt_color = '#797EF6'

    x_loc = 0
    starting_points = []
    for target in ssrt:

        if target in ssrt_empirical:
            hatch='/'
        else:
            hatch=None

        starting_points.append(x_loc)

        row = df[(df['target'] == target)]
        
        rects = ax[2].bar(x_loc, height=row['mean_scores_r2'], yerr=row['std_scores_r2'], 
                        width=.5, color=ssrt_color, hatch=hatch)
        x_loc += 1

    ax[2].grid(linestyle=':')
    ax[2].set_xticks(np.array(starting_points))
    ax[2].set_xticklabels(ssrt, rotation=45, ha='right')
    ax[2].set_title('SSRT')

    ax[2].spines[['top', 'left']].set_visible(False)

    # add blank space
    ax[3].axis('off')


    if save:
        plt.savefig(
            './data/08_reporting/vertexwise_regressor_modelfit.png', 
            bbox_inches='tight', 
            dpi=300
            )
        
def filter_contribution_scores(df, target_map: dict, threshold=.01, return_metric='avg_reg_coef'):

    targets_to_drop = ['gf.natural', 'tf.natural']

    df_fdr = mask_non_significant(df, threshold=threshold, return_metric=return_metric)
    df_fdr = df_fdr[~df_fdr['target'].isin(targets_to_drop)]

    df_fdr = df_fdr.replace(target_map)
    df = df.replace(target_map)
    
    return df_fdr, df

def _reindex_in(df):

    df = (df
          .drop(columns=['target', 'metric'])
          .set_index('condition')
        )
    
    return df

def _reindex_out(df):

    idx = ['target', 'condition']
    df = (df
          .reset_index()
          .set_index(idx)
          .reset_index()
        )
    
    return df


def separate_by_activation_direction(df: pd.DataFrame, df_subset: pd.DataFrame):

    avg_activation = df[df['metric'] == 'avg_activation']
    avg_activation = _reindex_in(avg_activation)
    
    targets = pd.unique(df['target'])

    pos = pd.DataFrame()
    neg = pd.DataFrame()

    for t in targets:
        # mask target for positive and negative
        tmp = df_subset[df_subset['target'] == t]
        tmp = _reindex_in(tmp)
       
        # mask values where condition is false (see pd.DataFrame.mask)
        p = tmp.mask(avg_activation < 0)
        n = tmp.mask(avg_activation >= 0)

        p['target'] = n['target'] = t
        pos = pd.concat([pos, p])
        neg = pd.concat([neg, n])

    pos, neg  = _reindex_out(pos), _reindex_out(neg)


    return pos, neg


def format_vertices(df: pd.DataFrame, metric:str, hemi:str) -> pd.DataFrame:
    '''Take BPt feature importance collection.
    Return a collection of arrays meant for Neurotools plotting.
    '''

    idx = ['target', 'condition']

    df = (df
        .set_index(idx)
        .filter(regex=hemi)
        .reset_index())

    df.columns = [c.replace(hemi + '_', '') for c in df.columns]


    df['name'] = df['target'] + '__' + df['condition']
    df = df.drop(columns=idx)

    df = df.T
    df.columns = df.iloc[-1]

    df = df.reset_index(names='vertex')
    df = df[df['vertex'] != 'name']
    df['vertex'] = df['vertex'].apply(pd.to_numeric)
    df.sort_values('vertex')

    tmp = pd.DataFrame({'vertex': np.arange(10242)})
    tmp = tmp[~tmp['vertex'].isin(df['vertex'])]

    tmp = df.merge(tmp, on='vertex', how='outer')
    tmp = tmp.sort_values('vertex')

    tmp = tmp.T
    tmp.columns = tmp.iloc[0]

    tmp = tmp.reset_index()
    tmp = tmp[tmp['index'] != 'vertex']

    out = tmp['index'].str.split('__', expand=True)
    out.columns = ['target', 'condition']

    tmp = tmp.drop(columns=['index'])
    tmp = tmp.assign(**out).set_index(['target', 'condition'])

    return tmp


def reshape_vertexwise(df):
    df = (df
    .reset_index().melt(id_vars=['target', 'condition'])
    .pivot(index=['condition', 'vertex'], columns='target')
    )

    df.columns = df.columns.to_flat_index()
    df.columns = [c[1] for c in df.columns]

    df = df.fillna(0)
    
    return df

def vertexwise_to_dict(df_hemi, target: str):
    hemi_dict = (df_hemi
            .groupby('condition')
            .apply(lambda x: x[target].values)
            .to_dict())

    return hemi_dict

def _make_collection(lh_reshaped, rh_reshaped):
    dict_collect = {}
    targets = list(lh_reshaped)

    for targ in targets:
        lh_dict = vertexwise_to_dict(lh_reshaped, targ)
        rh_dict = vertexwise_to_dict(rh_reshaped, targ)

        merged_lr = {}
        for k in lh_dict.keys():
            merged_lr[k] = [lh_dict[k], rh_dict[k]]
        
        dict_collect[targ] = merged_lr
    
    return dict_collect

def _make_plotting_group(df, metric):

    lh = format_vertices(df, hemi='lh', metric=metric)
    rh = format_vertices(df, hemi='rh', metric=metric)

    lh_reshaped = reshape_vertexwise(lh)
    rh_reshaped = reshape_vertexwise(rh)

    dict_collection = _make_collection(lh_reshaped, rh_reshaped)

    return dict_collection

def generate_plotting_collections(pos, neg, metric='avg_reg_coef'):
    pos_collection = _make_plotting_group(pos, metric)
    neg_collection = _make_plotting_group(neg, metric)

    return pos_collection, neg_collection




def make_posneg_plot(pos_lr, neg_lr, vmin, vmax, ax0, ax1, cmap=None):

    cmap = plt.cm.get_cmap('bwr')
    

    ntp.plot(
        pos_lr,
        vmin=vmin,
        vmax=vmax,
        ax=ax0,
        colorbar=False,
        threshold=0,
        cmap=cmap
    )

    ntp.plot(
        neg_lr,
        vmin=vmin,
        vmax=vmax,
        ax=ax1,
        colorbar=False,
        threshold=0,
        cmap=cmap
    )


def make_collage_plot(pos_collection, neg_collection, target, mode,
                      fontsize=25, full_vmin=None, full_vmax=None):    
    
    n_cond = 4
    width_ratios = [1]
    height_ratios = [100, 1, 100]

    col_ratios = list(repeat(10, n_cond))

    width_ratios.extend(col_ratios)
    width_ratios.extend([2]) # colorbar

    gs = {
        'width_ratios': width_ratios,
        'height_ratios': height_ratios,
        'hspace':0,
        'wspace':0
    }

    cmap = 'bwr'
    nb_ticks = 5
    cbar_tick_format='%.2g'

    fig, axs = plt.subplots(3, n_cond + 1+1, figsize = (55, 20) , gridspec_kw=gs)

    directions =  ['Positive\nActivation',"", 'Negative\nActivation']

    for i, direction in enumerate(directions):
        ax = axs[i, 0]
        ax.set_axis_off()
        if i==1: 
            continue
        else:
            ax.text(0, .5, direction, fontsize=fontsize)
        
    arbitrary_target = 'Go evidence threshold ($B$)'
    conditions = pos_collection[arbitrary_target].keys()
    vmin, vmax = full_vmin, full_vmax # plotting value max/min across conditions
  
    cnt = 1
    for condition in conditions:
        pos_lr = pos_collection[target][condition]
        neg_lr = neg_collection[target][condition]

        top = axs[0, cnt]
        middle = axs[1, cnt]
        bottom = axs[2, cnt]

        middle.set_axis_off() # make blank space

        make_posneg_plot(pos_lr, neg_lr, vmin=vmin, vmax=vmax, ax0=top, ax1=bottom, cmap=cmap)
        
        top.set_title(condition, fontsize=fontsize)
        cnt += 1
    
    # plot colorbar
    norm = Normalize(vmin=vmin, vmax=vmax)
    proxy_mappable = ScalarMappable(norm=norm, cmap=cmap)
    ticks = np.linspace(vmin, vmax, nb_ticks)

    right = axs[:, n_cond + 1]
    
    for ax in right.flat:
        ax.set_axis_off()
        
    cax, kw = make_axes(right, fraction=.5, shrink=0.5)
    cbar = fig.colorbar(proxy_mappable, cax=cax, ticks=ticks,
                        orientation='vertical', format=cbar_tick_format,
                        ticklocation='left')
    
    cbar.set_label(label='Avg. Regression Coef.', fontsize=fontsize - 2)

    prefix = 'Target: '
    title = target

    fig.suptitle(prefix + title, fontsize=fontsize+2)

    fname=target.lower().replace('.', '').replace(' ', '_').replace('$', '')
    # plt.savefig(f'./data/08_reporting/posneg_plots_{mode}/{fname}', dpi=300, bbox_inches='tight')
    plt.savefig(f'./data/08_reporting/posneg_plots_{mode}/{fname}.svg', bbox_inches='tight')

def make_compare_plot(pos_group, targets, conditions, fontsize=20, base_name='data/08_reporting/'):
    extremes = []
    for target, condition in zip(targets, conditions):
        pos_lr = pos_group[target][condition]

        for item in pos_lr:
            max = np.abs(item.max())
            min = np.abs(item.min())
            pmax = np.max([max, min])
            extremes.append(pmax)

    limit = np.max(np.array(extremes)) * 1.05 # add 5%
    vmin, vmax = -limit, limit

    n_cond = 2

    col_ratios = list(repeat(10, n_cond))

    width_ratios = col_ratios
    width_ratios.extend([2]) # colorbar
    gs = {
    'width_ratios': width_ratios,
    'hspace':0,
    'wspace':0
    }

    cmap = 'bwr'
    nb_ticks = 5
    cbar_tick_format='%.2g'

    fig, axs = plt.subplots(1, n_cond + 1, figsize = (30, 10) , gridspec_kw=gs)

    cnt = 0
    for target, condition in zip(targets, conditions):
        pos_lr = pos_group[target][condition]

        top = axs[cnt]
        cmap = plt.cm.get_cmap('bwr')


        ntp.plot(
            pos_lr,
            vmin=vmin,
            vmax=vmax,
            ax=top,
            colorbar=False,
            threshold=0,
            cmap=cmap
        )
        axtitle = target + "\n\n" + condition
        top.set_title(axtitle, fontsize=fontsize)
        cnt += 1

        norm = Normalize(vmin=vmin, vmax=vmax)
        proxy_mappable = ScalarMappable(norm=norm, cmap=cmap)
        ticks = np.linspace(vmin, vmax, nb_ticks)

        right = axs[n_cond]
        
        right.set_axis_off()
            
    cax, kw = make_axes(right, fraction=.5, shrink=0.5)
    cbar = fig.colorbar(proxy_mappable, cax=cax, ticks=ticks,
                        orientation='vertical', format=cbar_tick_format,
                        ticklocation='left')


    cbar.set_label(label='Avg. Regression Coef.', fontsize=fontsize - 4)
    plt.savefig(f'{base_name}compare_figure.png', dpi=300, bbox_inches='tight')
    # plt.savefig(f'{base_name}compare_figure.svg',bbox_inches='tight')

    



def generate_contribution_plots(df_fdr, pos_collection, neg_collection, mode='full'):

    targets = pd.unique(df_fdr['target'])
    
    for target in targets:
        full_vmin, full_vmax = get_fullrang_minmax(df_fdr, target=target)

        plt.clf()
        make_collage_plot(
            pos_collection, 
            neg_collection, 
            target=target, 
            full_vmax=full_vmax, 
            full_vmin=full_vmin,
            mode=mode
            )
        
        plt.clf()

def load_reg_coef(df):
    df = df[df['metric'] == 'avg_reg_coef']
    return df.drop(columns='metric')

def create_correlation_df(df: pd.DataFrame, columns: str, index: list) -> pd.DataFrame:

    df_long = df.melt(id_vars = ['target', 'condition'])
    df = df_long.pivot(columns=columns, index=index)

    df.columns = df.columns.to_flat_index()
    df.columns = [c[1] for c in df.columns]

    return df

def make_parameter_correlation_matrix(df, columns, index):

    drop_cond = ['correct_go_mrt', 'correct_go_rt', 'correct_go_stdrt', 'issrt']
    df = create_correlation_df(df, columns=columns, index=index)
    out = (df
        .reset_index()
        .set_index('variable')
    )

    if columns == 'target':
        out = out.drop(columns=drop_cond)
    elif columns == "condition":
        out = out[~out['target'].isin(drop_cond)]

    out = (out
        .groupby(index[0])
        .corr()
        .reset_index(names=[index[0], columns])
        .set_index(columns)
    )
    return out.groupby(index[0])


def make_brainmap_correlation_plot(df, columns, index, cmap='Spectral_r', 
                                   ncols=4, nrows=4, figsize=(25, 25)):
    grouped = make_parameter_correlation_matrix(df, columns, index)

    n_plots = len(grouped)

    if n_plots % 2 == 0:
        ncols = nrows = int(n_plots / 2)

    n_grid = ncols * nrows

    vmin, vmax = -1, 1
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)

    for g, ax in zip(grouped, axs.flat):
        condition = g[0]
        to_plot = g[1].drop(columns=index[0])
        
        sns.heatmap(
            to_plot, 
            vmin=vmin,
            vmax=vmax, 
            cmap=cmap,
            cbar=False, 
            annot=True,
            square=True,
            ax=ax
        )
        ax.set_ylabel('')
        ax.set_title(condition)

    # make blank subplots if size of grid > # plots
    for i in range(n_plots, n_grid):
        axs.flat[i].axis('off')

    fname = f"data/08_reporting/{columns}_by_{index[0]}_correlations.png"
    plt.savefig(fname, bbox_inches='tight', dpi=300)


def corrfunc(x, y, ax=None, **kws):
    r, p = pearsonr(x, y)
    ax = ax or plt.gca()
    if p<.001:
        ax.annotate(f'r = {r:.2f}, p < .001', xy=(.4, .9), xycoords=ax.transAxes)
    else:
        ax.annotate(f'r = {r:.2f}, p < {p:.3f}', xy=(.4, .9), xycoords=ax.transAxes)

def make_parameter_correlation_plot(df):
    sns.set_context("paper", rc={"font.size":20,"axes.titlesize":20,"axes.labelsize":20})   
    drop_cols = ['tf.natural', 'gf.natural']
    df= df.drop(columns=drop_cols)

    g = sns.PairGrid(df, corner=True, diag_sharey=False)
    g = g.map_lower(sns.regplot, scatter_kws={'alpha': 0.01})
    g = g.map_diag(sns.histplot)
    g = g.map_lower(corrfunc)

    plt.savefig('data/08_reporting/parameter_correlations.png', dpi=150, bbox_inches='tight')

def _get_brain_vector(df, condition, target):

    idx = ['condition', 'target']
    df = (df[(df['condition'] == condition) & 
        (df['target'] == target)]
            .set_index(idx)
            .melt()
            .drop(columns='variable')
            .rename(columns={'value':f'{condition}_{target}'})
    ) 
    return df

# def get_two_parameter_brainmaps(df, condition1='correct_go', condition2='correct_stop',
#                                 target1='EEA', target2='tf',
#                                 names=['EEA: Correct Go', '$P_{tf}$: Correct Stop']):
    
#     df1 = _get_brain_vector(df, condition1, target1)
#     df2 = _get_brain_vector(df, condition2, target2)

#     out = pd.concat([df1, df2], axis=1)
#     out.columns = names

#     return out



def get_two_parameter_brainmaps(df, condition1='Correct Go', condition2='Correct Stop',
                                target1='Efficiency of evidence acc. (EEA)',
                                  target2='Probability of trigger failure ($p_{tf}$)',
                                names=['EEA: Correct Go', '$P_{tf}$: Correct Stop']):
    
    df1 = _get_brain_vector(df, condition1, target1)
    df2 = _get_brain_vector(df, condition2, target2)

    out = pd.concat([df1, df2], axis=1)
    out.columns = names

    return out

def make_two_parameter_correlation_plot(df_maps, df_brain, alpha=0.01,
                                        fpath='./data/08_reporting/'):

    df_brain = get_two_parameter_brainmaps(df_brain)
    df_brain[df_brain == 0] = np.NaN
    df_brain = df_brain.dropna()

    scatter_kws = {'alpha': alpha}
    fig, axs = plt.subplots(1,2, figsize=(7, 3))
    ax = axs[0]
    sns.regplot(
        data=df_maps,
        x='tf',
        y='EEA',
        scatter_kws=scatter_kws,
        color="#440154FF",
        ax=ax
    )
    corrfunc(df_maps['tf'], df_maps['EEA'], ax=ax)
    ax.set_xlabel('$P_{tf}$')
    ax.set_title('Parameter Correlation')

    ax = axs[1]
    x = df_brain.iloc[:, 1] 
    y = df_brain.iloc[:, 0]
    sns.regplot(
        data=df_brain,
        x=x,
        y=y,
        scatter_kws=scatter_kws,
        color="#22A884FF",
        ax=ax
    )
    corrfunc(x, y, ax=ax)
    ax.set_title('Brain Map Correlation')

    for ax in axs:

        ax.tick_params(left=False, bottom=False)
        ax.set_xticks([])
        ax.set_yticks([])

    fpath = f'{fpath}tf_eea_correlation.png'
    plt.savefig(fpath, bbox_inches='tight', dpi=300)



