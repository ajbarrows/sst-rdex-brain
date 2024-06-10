import pandas as pd
import numpy as np

def load_confounds(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df[df['eventname'] == 'baseline_year_1_arm_1']
    df = df[['subjectkey', 'abcd_site', 'age', 'female']]
    
    return df.set_index('subjectkey')

def load_weigard(df: pd.DataFrame) -> pd.DataFrame:
    '''Load RDEX-ABCD parameter estimates (e.g., maximum a posteriori).'''

    if df.index.name != 'subjectkey':
        df.index.name = 'subjectkey'

    df = df.drop(columns=['k', 'tf.natural', 'gf.natural'])

    return df

def load_dk_ssrt(df: pd.DataFrame) -> pd.DataFrame:
    '''Load empirically-derrived SSRT values.'''

    df = df[df['eventname'] == 'baseline_year_1_arm_1']
    df = df.rename({'subject': 'subjectkey', 'ssrt': 'SSRT_dk'}, axis=1)
    df['SSRT_dk'] = df['SSRT_dk'] / 1000
    df = df.drop('eventname', axis=1)
    
    return df.set_index('subjectkey')

def merge_predictors(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:

    df = (df1
          .reset_index()
          .set_index('subjectkey')
          .join(df2.reset_index().set_index('subjectkey'))
    )

    return df

def load_predictors(rdex: pd.DataFrame, e_ssrt: pd.DataFrame) -> pd.DataFrame:
    
    rdex = load_weigard(rdex)
    e_ssrt = load_dk_ssrt(e_ssrt)

    pred = merge_predictors(rdex, e_ssrt)

    return pred

def load_phenotype(pheno: pd.DataFrame) -> pd.DataFrame:

    return pheno.set_index('subjectkey')


def load_just_rdex(df: pd.DataFrame) -> pd.DataFrame:

    excl_vars = ['correct_go_mrt', 'correct_go_stdrt', 'correct_go_rt', 'issrt']

    return df.drop(columns=excl_vars)

def load_just_empirical(df: pd.DataFrame) -> pd.DataFrame:
    
    incl_vars = ['correct_go_mrt', 'correct_go_stdrt', 'correct_go_rt', 'issrt']

    return df[incl_vars]