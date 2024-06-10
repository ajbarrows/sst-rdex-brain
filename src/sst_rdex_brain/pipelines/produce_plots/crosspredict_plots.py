import pandas as pd
import BPt as bp
import matplotlib.pyplot as plt
import seaborn as sns


def transform_crosspredict_features(res: bp.CompareDict, process_map: dict, 
                                    target_map: dict) -> pd.DataFrame:
    
    for l, m in res.items(): 
        fis = m.get_fis()

    fis_long = fis.melt()

    # human-readable names
    fis_long['process'] = fis_long['variable']

    fis_long['process'] = fis_long['process'].replace(process_map)
    fis_long['variable'] = fis_long['variable'].replace(target_map)

    # sort by mean FIS
    fis_avg = fis_long.groupby('variable').mean()
    fis_avg.columns = ['mean']
    fis_plot = (fis_long
                .join(fis_avg, on='variable')
                .sort_values('mean')
    )

    return fis_plot


def make_crosspredict_plot(fis_plot_df: pd.DataFrame):

    palette = {
        'Go Process': '#77DD77',
        'Stop Process': "lightcoral"
    }

    fig, ax = plt.subplots()

    sns.barplot(data=fis_plot_df, y='variable', x='value',
                    hue='process', palette=palette,
                     dodge=False, ax=ax)
    ax.set_title('Feature Importance Predicting Empirical SSRT')
    ax.set_xlabel('Avg. Regression Coef.')
    ax.set_ylabel('')

    plt.legend(title='', loc='upper right')

    plt.savefig('./data/08_reporting/crosspredict_fis.png', dpi=300, bbox_inches='tight')