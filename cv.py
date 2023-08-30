import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

'''
@vencerlanz09
Displays 16 random pictures with their categories (in a 4x4 layout) 
'''
def display_16(image_df):
    # Display 16 picture of the dataset with their labels
    random_index = np.random.randint(0, len(image_df), 16)
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10), subplot_kw={'xticks': [], 'yticks': []})

    for i, ax in enumerate(axes.flat):
        ax.imshow(plt.imread(image_df.Filepath[random_index[i]]))
        ax.set_title(image_df.Label[random_index[i]])
    plt.tight_layout()
    plt.show()

'''
@vencerlanz09
Bar plot the 20 most frequent categories
'''
def most_freq_20(image_df):
    # Get the top 20 labels
    label_counts = image_df['Label'].value_counts()[:20]

    plt.figure(figsize=(20, 6))
    sns.barplot(x=label_counts.index, y=label_counts.values, alpha=0.8, palette='dark:salmon_r')
    plt.title('Distribution of Top 20 Labels in Image Dataset', fontsize=16)
    plt.xlabel('Label', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45)
    plt.show()

