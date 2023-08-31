# Comment out different sections as needed (e.g. if not doing CV, comment out CV imports)

'''======================= BASE (frames, math, plot) ============================'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')

'''=================== FILE MANIP. ==============================='''
import os
import random

'''================== COMPUTER VISION ======================='''
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.utils import image_dataset_from_directory, load_img
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Activation, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

'''=================== OPTIONAL ========================='''
from warnings import filterwarnings
filterwarnings('ignore')


'''===================== CONSTANTS (change as needed) ===================='''
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3
BATCH_SIZE = 32
EPOCHS = 15