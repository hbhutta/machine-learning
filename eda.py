import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


'''
Simple scatter plot with feature on x-axis and target on y-axis
'''
def plot_(df, feature: str, target: str):
    fig, ax = plt.subplots()
    ax.scatter(df[feature], df[target])
    plt.ylabel(target, fontsize=13)
    plt.xlabel(feature, fontsize=13)
    plt.show()



'''
Bar plot percentage of missing data for each feature
'''
def plot_percent_missing(train_df, test_df, feature: str):

    all_data = pd.concat((train_df, test_df)).reset_index(drop=True)
    all_data.drop([feature], axis=1, inplace=True)

    all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
    missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

    f, ax = plt.subplots(figsize=(15, 12))
    plt.xticks(rotation='vertical')
    sns.barplot(x=all_data_na.index, y=all_data_na)
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent of missing values', fontsize=15)
    plt.title('Percent missing data by feature', fontsize=15)



'''
Returns a summary of the given dataframe.

This function captures most of what's usually done in preprocessing, such as:
- Quantity and ratio of missing values per feature
- Min and max
- First few values in the head

[Credits to @datamanyo]
'''
def summary(df):
    print(f'data shape: {df.shape}')
    summ = pd.DataFrame(df.dtypes, columns=['data type'])
    summ['#missing'] = df.isnull().sum().values * 100
    summ['%missing'] = df.isnull().sum().values / len(df)
    summ['#unique'] = df.nunique().values
    desc = pd.DataFrame(df.describe(include='all').transpose())
    summ['min'] = desc['min'].values
    summ['max'] = desc['max'].values
    summ['first value'] = df.loc[0].values
    summ['second value'] = df.loc[1].values
    summ['third value'] = df.loc[2].values
    
    return summ


