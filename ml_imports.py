

'''==================================== BASE: Math, vis, dataframes ================================='''
import pandas as pd
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) # limiting floating point to 3 decimal places

# Visualization
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')


import matplotlib.pyplot as plt
# %matplotlib inline # If this doesn't work comment it out


# Math (linear algebra), stats (box-cox transformation, normalization)
import numpy as np
from scipy import stats
'''
If doing regression, we need to ensure our data is normalized,
so we'll need some scipy packages.

We can visually check normalization with a quantile-quantile plot,
or by looking at a bar plot.

Different methods like box-cox or log transformations will work.
'''
from scipy.stats import norm, skew

'''================================== MODEL: Scaling, creating, evaluation ================================='''


# Scaling
from sklearn.preprocessing import RobustScaler
from mlxtend.preprocessing import min_max_scaling


# Validation framework
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold

# Metrics
'''
- roc_curve and roc_auc_score typically for classification
- mean_squared_error is the most common metric for standard regression
'''
from sklearn.metrics import mean_squared_error

# Preprocessing


'''
If your data has categorical variables, you have to numerically encode 
them because that's all the model accepts.

In the Kaggle tutorial, it's written that one-hot encoding usually performs 
the best, but we can always try out different things and compare.
'''
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder

'''
make_pipeline is the function alternative to instantiating a Pipeline instance,
so import one or the other by use case.
'''
from sklearn.pipeline import make_pipeline, Pipeline

'''
Transformations and imputations are typically a part of a pipeline
'''
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


# Model(s)
from sklearn.linear_model import LogisticRegression, ElasticNet, Lasso,  BayesianRidge

# from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone


# Ensembling and boosting
'''
Ensembling and boosting is an important step before model evaluation.
'''
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, StackingRegressor, RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

import warnings
warnings.filterwarnings('ignore')
RANDOM_STATE = 42 # global random state used when needed


