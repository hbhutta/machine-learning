import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')



# System libraries
from pathlib import Path
import os.path
import random


# Optional
import warnings
warnings.filterwarnings("ignore") # to clean up output cells

'''
Set seed for reproducibility
'''
def set_seed(seed=31415):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #os.environ['TF_DETERMINISTIC_OPS'] = '1'

'''
@vencerlanz09
Create and returns a pandas dataframe from a directory of images.
The directory can contain subdirectories.
'''
def images_to_df(dataset: str):
    image_dir = Path(dataset)

    # Get filepaths and labels - Accounting for different image file extensions
    filepaths = list(image_dir.glob(r'**/*.JPG')) + list(image_dir.glob(r'**/*.jpg')) + list(image_dir.glob(r'**/*.png')) + list(image_dir.glob(r'**/*.png'))

    labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

    filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    # Concatenate filepaths and labels
    image_df = pd.concat([filepaths, labels], axis=1)

    return image_df


'''
@vencerlanz09
Displays 16 random pictures with their categories (in a 4x4 layout).
'''
def display_9(image_df):
    # Display 9 picture of the dataset with their labels
    random_index = np.random.randint(0, len(image_df), 9)
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10), subplot_kw={'xticks': [], 'yticks': []})

    for i, ax in enumerate(axes.flat):
        ax.imshow(plt.imread(image_df.Filepath[random_index[i]]))
        ax.set_title(image_df.Label[random_index[i]])
    plt.tight_layout()
    plt.show()


'''
@vencerlanz09
Bar plot the 20 most frequent categories.
'''
def most_freq_20(image_df):
    # Get the top 20 labels
    label_counts = image_df['Label'].value_counts()[:20]

    plt.figure(figsize=(20, 6))
    sns.barplot(x=label_counts.index, y=label_counts.values, alpha=0.8, palette='#dd7700')
    plt.title('Distribution of Top 20 Labels in Image Dataset', fontsize=16)
    plt.xlabel('Label', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45)
    plt.show()


'''
@hbhutta
Plot training and validation loss curves given model history

Adapted from the code in the data augmentation exercise in the Kaggle tutorial
'''
def plot_loss_curves(history):
    history_frame = pd.DataFrame(history.history)
    history_frame.head()
    history_frame.loc[:, ['loss', 'val_loss']].plot()
    history_frame.loc[:, ['accuracy', 'val_accuracy']].plot();



'''
@vencerlanz09
Display model's predictions given test data and predictions
'''
def display_predictions(test_df, pred):
    random_index = np.random.randint(0, len(test_df) - 1, 15)
    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(25, 15),
                            subplot_kw={'xticks': [], 'yticks': []})

    for i, ax in enumerate(axes.flat):
        ax.imshow(plt.imread(test_df.Filepath.iloc[random_index[i]]))
        if test_df.Label.iloc[random_index[i]] == pred[random_index[i]]:
            color = "green"
        else:
            color = "red"
        ax.set_title(f"True: {test_df.Label.iloc[random_index[i]]}\nPredicted: {pred[random_index[i]]}", color=color)
    plt.show()
    plt.tight_layout()




