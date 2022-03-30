import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Calculated the missing rate of a column
def filling_factor(df):
    '''
    Show filling percentage of each variable
    '''
    missing_df = df.isnull().sum(axis=0).reset_index()
    missing_df.columns = ['column_name', 'missing_count']
    missing_df['filling_factor'] = (df.shape[0]-missing_df['missing_count'])/df.shape[0]*100
    missing_df = missing_df.sort_values('filling_factor').reset_index(drop = True)
    return missing_df

def barchart_percent(x, data, figsize=(12, 8), rotation=False):
    '''
    Plot barchart with text value
    '''
    plt.figure(figsize=figsize)
    ax = sns.countplot(x=x,
                        data=data)
    total = len(data)
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.1f}%\n'
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax.annotate(percentage, (x, y), ha='center', va='center')
        
    if rotation == True:
        plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def EDA_continous(data):
    '''
    Univariate visualisation of continous varibale
    '''
    for col in data.columns:
        fig = plt.figure(figsize=(20, 5))

        plt.subplot(1, 2, 1)
        sns.histplot(x=data[col],
                     data=data,
                     kde=True)

        plt.subplot(1, 2, 2)
        sns.boxplot(x=data[col],
                    data=data)
        fig.tight_layout(pad=0.8)

        plt.show()
