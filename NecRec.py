#Librerias
import timeit
import pandas as pd
import numpy as np
import math
import re
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
#from surprise import Reader, Dataset, SVD, evaluate
from surprise import Reader
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate

#cargar datos
start = timeit.timeit()
df1 = pd.read_csv('combined_data_1.txt', header = None, names = ['Cust_Id', 'Rating'], usecols = [0,1])
#df2 = pd.read_csv('combined_data_2.txt', header = None, names = ['Cust_Id', 'Rating'], usecols = [0,1])
"""df3 = pd.read_csv('combined_data_3.txt', header = None, names = ['Cust_Id', 'Rating'], usecols = [0,1])
df4 = pd.read_csv('combined_data_4.txt', header = None, names = ['Cust_Id', 'Rating'], usecols = [0,1])"""

df1['Rating'] = df1['Rating'].astype(float)
#df2['Rating'] = df2['Rating'].astype(float)
"""df3['Rating'] = df3['Rating'].astype(float)
df4['Rating'] = df4['Rating'].astype(float)"""

"""print('Dataset 1 shape: {}'.format(df1.shape))
print('-Dataset examples-')
print(df1.iloc[::5000000, :])
print('Dataset 2 shape: {}'.format(df2.shape))
print('-Dataset examples-')
print(df2.iloc[::5000000, :])
print('Dataset 3 shape: {}'.format(df3.shape))
print('-Dataset examples-')
print(df3.iloc[::5000000, :])
print('Dataset 4 shape: {}'.format(df4.shape))
print('-Dataset examples-')
print(df4.iloc[::5000000, :])"""