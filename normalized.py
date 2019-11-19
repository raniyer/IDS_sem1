#Data Preprocessing

#importing libraries
import numpy as np
import re
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing

#importing dataset
dataset = pd.read_csv('master.csv')
dataset.drop(["country-year", "HDI for year", "generation"], axis = 1, inplace = True)

# label encoding the data 
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder() 
dataset['country']= le.fit_transform(dataset['country']) 
dataset['sex']= le.fit_transform(dataset['sex'])
dataset['age']= le.fit_transform(dataset['age'])
dataset[' gdp_for_year ($) '] = dataset[" gdp_for_year ($) "].str.replace(',', '')
dataset[' gdp_for_year ($) '] = dataset[' gdp_for_year ($) '].astype(Long)
type(dataset[' gdp_for_year ($) '])
dataset.dtypes [' gdp_for_year ($) ']
# normalize the data attributes

normalized = preprocessing.normalize(dataset)
