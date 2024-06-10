import numpy as np
import pandas as pd
import BPt as bp

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import ElasticNet
from BPt.extensions import LinearResidualizer
from statsmodels.api import OLS

from typing import Any, Callable, Dict, List

def prepare_dataset(df: pd.DataFrame, targets: list, scopes: dict) -> bp.Dataset:
    
    df = df[~df.index.duplicated(keep='first')]

    ds = bp.Dataset(df, targets=targets)
    for k, v in scopes.items():
         ds.add_scope(v, k, inplace=True)

    ds = ds.auto_detect_categorical()
    ds.add_scope('mri_info_deviceserialnumber', 'category', inplace=True)
    ds = ds.ordinalize('category')

    ds.dropna(inplace=True)

    ds = ds.set_test_split(.2, random_state=42)

    return ds

def prepare_dataset_crosspredict(df: pd.DataFrame, targets: list) -> bp.Dataset:

    df = df[~df.index.duplicated(keep='first')]

    ds = bp.Dataset(df, targets=targets)

    ds.dropna(inplace=True)

    ds = ds.set_test_split(.2, random_state=42)

    return ds


def residualize_select(df: pd.DataFrame, residualize_for: list, targets_to_resid: list) -> pd.DataFrame:
    '''Create new targets as (linearly) residualized versions of existing targets.'''

    X = df[residualize_for]
    targets = df[targets_to_resid]

    residual_targets = pd.DataFrame()
    for c in targets:
        y = targets[c].values
        ols_model = OLS(y, X).fit()

        lab = c + '_resid'
        residual_targets[lab] = ols_model.resid

    scopes = list(residual_targets)

    return residual_targets, scopes

def define_regression_pipeline(ds: bp.Dataset, scopes: Dict[str, List[str]], model: str) -> bp.Pipeline:
    
    # Just scale float type features
    scaler = bp.Scaler('robust', scope='float')

    # Define residualization procedure
    ohe = OneHotEncoder(categories='auto', drop='if_binary', sparse=False, handle_unknown='infrequent_if_exist')
    ohe_tr = bp.Transformer(ohe, scope='category')
    resid = LinearResidualizer(to_resid_df=ds['covariates'], fit_intercept=True)
    resid_tr = bp.Scaler(resid, scope=list(scopes.keys()))

    # Define regression model
    mod_params = {'alpha': bp.p.Log(lower=1e-5, upper=1e5)}

    if model == "ridge":
        mod_obj="ridge"
    elif model == 'elastic':
        mod_obj=ElasticNet()
        l1_ratio = bp.p.Scalar(lower=0.001, upper=1).set_mutation(sigma=0.165)
        mod_params['l1_ratio'] = l1_ratio

    param_search = bp.ParamSearch('HammersleySearch', n_iter=100, cv='default')
    model = bp.Model(
        obj=mod_obj, 
        params=mod_params,  
        param_search=param_search
    )

    # Then define full pipeline
    pipe = bp.Pipeline([scaler, ohe_tr, resid_tr, model])

    return pipe


def fit_map_model(ds: bp.Dataset, scopes: Dict[str, List[str]], model='ridge', complete=True) -> bp.CompareDict:
    
    if model == 'elastic' or model == 'ridge':
        pipe = define_regression_pipeline(ds, scopes, model=model)
    else:
        raise Exception(f'Specificed model {model} is not implemented')

    cv = bp.CV(splits=5, n_repeats=1)
    ps = bp.ProblemSpec(n_jobs=64, random_state=42)
    
    if not complete:
        compare_scopes = []
        for key in scopes.keys():
            if key != 'covariates':
                compare_scopes.append(bp.Option(['covariates', key], name=f'cov + {key}'))
            else:
                compare_scopes.append('covariates')
        compare_scopes = bp.Compare(compare_scopes)
    else:
        compare_scopes = None

    results = bp.evaluate(pipeline=pipe,
                      dataset=ds,
                      problem_spec=ps,
                      scope=compare_scopes,
                      mute_warnings=True,
                      target=bp.Compare(ds.get_cols('target')),
                      cv=cv)

    return results

def define_crosspredict_pipeline(ds: bp.Dataset, scopes: Dict[str, List[str]]) -> bp.Pipeline:
   
    # Just scale float type features
    scaler = bp.Scaler('robust', scope='float')
    normalizer = bp.Scaler('normalize', scope='float')

    # Define regression model
    mod_params = {'alpha': bp.p.Log(lower=1e-5, upper=1e5)}


    mod_obj='ridge'

    param_search = bp.ParamSearch('HammersleySearch', n_iter=100, cv='default')
    model = bp.Model(
        obj=mod_obj, 
        params=mod_params,  
        param_search=param_search
    )

    # Then define full pipeline
    pipe = bp.Pipeline([scaler, normalizer, model])

    return pipe


def fit_crosspredict_model(ds: bp.Dataset, scopes: Dict[str, List[str]]) -> bp.CompareDict:

    pipe = define_crosspredict_pipeline(ds, scopes)
    cv = bp.CV(splits=5, n_repeats=1)
    ps = bp.ProblemSpec(n_jobs=8, random_state=42)


    results = bp.evaluate(pipeline=pipe,
                      dataset=ds,
                      problem_spec=ps,
                      mute_warnings=True,
                      target=bp.Compare(ds.get_cols('target')),
                      cv=cv)

    return results

def run_permutation_tests(results: bp.CompareDict) -> list:

    p_values = {}
    p_scores = {}

    for k, v in results.items():

        lab = k.__dict__['options']

        target = lab[0].__dict__['name']

        
        pvals, pscores = v.run_permutation_test()

        p_values[target] = pvals
        p_scores[target] = pscores

    
    out = {'p_values': p_values, 'p_scores': p_scores}

    return out


def print_avg_effect(df: pd.DataFrame):
    
    targets = df['target']

    for t in targets:

        mean_r2 = df[df['target'] == t]['mean_scores_r2'][0]
        std_r2 = df[df['target'] == t]['std_scores_r2'][0]

        print(f'Target: {t} \t Avg. R^2 = {mean_r2} +/- {std_r2}')