import pandas as pd
import numpy as np


def load_behavioral(df: pd.DataFrame) -> pd.DataFrame:
    
    idx = ['src_subject_id', 'eventname']

    return df.set_index(idx)


def assemble_sst_behavioral(sst_beh:pd.DataFrame, conditions:dict, rt_vars=None) -> pd.DataFrame:

    sub = sst_beh.filter(regex='all|filter')

    out = (sub
            .filter(rt_vars.keys())
            .rename(columns=rt_vars)
        )
    
    cond = conditions.copy()
    pop_items = ['_go', 'beh_s']
    incl_str = ['_nt', '_rt']

    for s in incl_str:

        if s == '_rt':
            for itm in pop_items:
                cond.pop(itm)

        for k, v in cond.items():
            filter_string = k + s
            tmp = sub.filter(like=filter_string)
            tmp.columns = [v + s]

            out = pd.concat([out, tmp], axis=1)
    

    return out

def flag_uvm_sst_exclusion(df: pd.DataFrame) -> pd.DataFrame:

    #TODO pleace in config file

    GO_COUNT_MIN=300
    STOP_COUNT_MIN=30
    
    CORRECT_GO_RATE_MIN=0.6
    INCR_GO_RATE_MAX=0.3
    LATEGO_RATE_MAX=0.3 # across correct and incorrect latego
    NORESPONSE_GO_RATE_MAX=0.3
    CORRECT_STOP_RATE_MIN=0.2
    CORRECT_STOP_RATE_MAX=0.8

    MEAN_SSRT_MIN=50

    df['uvm_sst_beh_exclude'] = np.where(
        (df['go_nt'] < GO_COUNT_MIN)
        | (df['stop_nt'] < STOP_COUNT_MIN)
        | (df['correct_go_rt'] < CORRECT_GO_RATE_MIN)
        | (df['incorrect_go_rt'] > INCR_GO_RATE_MAX)
        | (df['correct_latego_rt'] + df['incorrect_latego_rt'] > LATEGO_RATE_MAX)
        | (df['noresponse_go_rt'] > NORESPONSE_GO_RATE_MAX)
        | (df['correct_stop_rt'] < CORRECT_STOP_RATE_MIN)
        | (df['correct_stop_rt'] > CORRECT_STOP_RATE_MAX)
        | (df['mssrt'] < MEAN_SSRT_MIN),
        True,
        False
    )

    return df

def flag_sst_mri_exclusion(df: pd.DataFrame, mri_qc: pd.DataFrame) -> pd.DataFrame:

    idx = ['src_subject_id', 'eventname'] 

    if not all([i in mri_qc.index for i in idx]):
        mri_qc = mri_qc.set_index(idx)

    mri_qc = (mri_qc
              .filter(['imgincl_sst_include'])
              .rename(columns={'imgincl_sst_include':'nda_sst_mri_exclude'})
              .replace({1: False, 0: True}) # recode to exclude=True
    )

    df = df.join(mri_qc) 
    
    df['nda_sst_mri_exclude'] = df['nda_sst_mri_exclude'].astype('bool')
    
    return df

def flag_sst_beh_exclusion(df: pd.DataFrame, sst_beh: pd.DataFrame) -> pd.DataFrame:
    sst_beh = (sst_beh
                .filter(['tfmri_sst_beh_performflag'])
                .replace({1: False, 0: True}) # recode to exclude=True
                .rename(columns={'tfmri_sst_beh_performflag':'nda_sst_beh_exclude'})
      )

    return df.join(sst_beh)

def subset_baseline(sst_metrics: pd.DataFrame) -> pd.DataFrame:

    sst_metrics_bl = (sst_metrics
                        .query("eventname == 'baseline_year_1_arm_1'")
                        .drop(columns='eventname')
                        .set_index('src_subject_id')
                    )
    
    return sst_metrics_bl

def apply_exclusion_criteria(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.query("~nda_sst_mri_exclude")

    return df   

def abcd5_loader(partitioned_data, keywords:tuple) -> pd.DataFrame:
    '''
    Load any (we hope) file in the ABCD 5.0+ release using
    keywords that appear in the filename.
    '''
    
    idx = ['src_subject_id', 'eventname']
    df = pd.DataFrame(columns=idx).set_index(idx)

    keywords = tuple(keywords)

    for dataset, loader in partitioned_data.items():
        if any(k in dataset for k in keywords):
            tmp = loader()
            df = df.join(tmp.set_index(idx), how='outer')
        
    return df


def assemble_phenotype_df(df: pd.DataFrame, *var_dicts) -> pd.DataFrame:

    phenotype = pd.DataFrame()

    for var in var_dicts:

        tmp = df.filter(var.keys())
        tmp = tmp.rename(columns=var)

        phenotype = pd.concat([phenotype, tmp], axis=1)
        
    return phenotype.reset_index()

def merge_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    return df1.join(df2)


def filter_phenotype(phenotype: pd.DataFrame, map: pd.DataFrame, tpt: str):
    '''Limit phenotype dataset to subjects available in RDEX data'''

    phenotype = phenotype[phenotype['eventname'] == tpt]
    phenotype = (phenotype
             .drop(columns='eventname')
             .rename(columns={'src_subject_id': 'subjectkey'})
             .set_index('subjectkey')
    )

    df = map.join(phenotype)
    df = df.dropna(axis=1, how='all').dropna()
    
    rdex = df[list(map)]
    pheno = df.drop(columns=list(map))

    return rdex, pheno