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

#Combinar los 4 datasets
df = df1
#df = df1.append(df2)
"""df = df.append(df3)
df = df.append(df4)"""

df.index = np.arange(0, len(df))
print('Full dataset shape: {}'.format(df.shape))
print('-Dataset examples-')
print(df.iloc[::5000000,:])

#Veamos primero cómo se propagan los datos:
p = df.groupby('Rating')['Rating'].agg(['count'])
#conteo de las peliculas
movie_count = df.isnull().sum()[1]
#conteo de los usuarios
cust_count = df['Cust_Id'].nunique() - movie_count
#conteo de los ratings
rating_count = df['Cust_Id'].count() - movie_count

ax = p.plot(kind = 'barh', legend = False, figsize = (15,10))
plt.title('Total pool: {:,} Movies, {:,} customers, {:,} ratings given'.format(movie_count, cust_count, rating_count), fontsize=20)
plt.axis('off')

for i in range(1,6):
    ax.text(p.iloc[i-1][0]/4, i-1, 'Rating {}: {:.0f} %'.format(i, p.iloc[i-1][0]*100 / p.sum()[0]), color = 'white', weight = 'bold')