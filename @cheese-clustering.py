#!/usr/bin/env python
# coding: utf-8

# **Created by Gael AHOUANVOEDO**
# 
# **🚀Clustering of cheese -📊EDA + Modelisation📈**
# 
# **24 January 2022**
# 

# # <center> 🚀CLUSTERING OF CHEESE -📊EDA + MODELISATION📈 </center>
# ## <center>For more information you can contact me at +221781203911👍</center>

# # Table of Contents
# <a id="toc"></a>
# - [1. Introduction](#1)
# - [2. Imports](#2)
# - [3. Data Loading and Preperation](#3)
#     - [3.1 Exploring Dataset](#3.1)
# - [4. EDA](#4)
#     - [4.1 Overview of Data](#4.1)
#     - [4.2 Feature Distribution of Continous Features](#4.2)
#     - [4.7 Map of correlations ](#4.6)
# - [5. Data Pre-Processing](#5)    
# - [6. Modeling](#6)
#     - [6.1 YellowBrick](#6.1)
# - [7. Submission](#7)   

# <a id="1"></a>
# # **<center><span style="color:#fd7b12;">Introduction  </span></center>**

# **This project is for my personnal `Portfolio` it is not a delivery program `Just to show off my skills ` Thanks.**
# 
# 
# **In this one I am supposed to claclustering  various varieties of cheeses into a set n to be determined.**
# 
# **Submissions are evaluated on `Optimal cluster number`.**

# <a href="#toc" role="button" aria-pressed="true" >⬆️Back to Table of Contents ⬆️</a>

# <a id="2"></a>
# # **<center><span style="color:#fd7b12;">Imports  </span></center>**

# ### <span style="color:#e76f51;"> Useful libraries : </span>

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from yellowbrick.cluster import KElbowVisualizer
import warnings
warnings.filterwarnings('ignore')


# <a href="#toc" role="button" aria-pressed="true" >⬆️Back to Table of Contents ⬆️</a>

# <a id="3"></a>
# # **<center><span style="color:#fd7b12;">Data Loading and Preparation </span></center>**

# In[2]:


data = pd.read_table('fromage.txt')


# In[3]:


df = pd.read_table('fromage.txt', index_col=0)


# In[4]:


df.head()


# ## <span style="color:#e76f51;"> Column Descriptions  : </span>
# 
# 
# - `Calories` - calorie content in a unit of cheese.
# - `Sodium` - sodium content in a unit of cheese.
# - `Calcium` - calcium content in a unit of cheese.
# - `Lipides` - lipides content in a unit of cheese.
# - `Retinol` - retinol content in a unit of cheese.
# - `Folates` - folates content in a unit of cheese.
# - `Proteines` - proteines content in a unit of cheese.
# - `Cholesterol` - cholesterol content in a unit of cheese.
# - `Magnesium` - magnesium content in a unit of cheese.
# 

# <a id="3.1"></a>
# ## <span style="color:#e76f51;"> Exploring Dataset : </span>

# <div class="alert alert-block alert-info" style="font-size:14px; font-family:verdana; line-height: 1.7em;">
#     📌 &nbsp;<b><u>Observations in Dataset:</u></b><br>
#  
# * <i> There are total of <b><u>9</u></b> columns and <b><u>29</u></b> rows in <b><u>the dataset.</u></b></i><br>
# * <i> Dataset contains <b><u>261</u></b> observation with <b><u>0</u></b>  missing values.</i><br>

# ### <span style="color:#e76f51;"> Quick view of Train Data : </span>

# Below are the first 5 rows of the dataset:

# In[5]:


df.head()


# In[6]:


print(f'\033[94mNumber of rows in the dataset: {df.shape[0]}')
print(f'\033[94mNumber of columns in the dataseta: {df.shape[1]}')
print(f'\033[94mNumber of values in the dataset: {df.count().sum()}')
print(f'\033[94mNumber missing values in the dataset: {sum(df.isna().sum())}')


# ### <span style="color:#e76f51;"> Basic statistics of the dataset : </span>

# Below is the basic statistics for each variables which contain information on `count`, `mean`, `standard deviation`, `minimum`, `1st quartile`, `median`, `3rd quartile` and `maximum`.

# In[7]:


df.describe()


# <a href="#toc" role="button" aria-pressed="true" >⬆️Back to Table of Contents ⬆️</a>

# <a id="4"></a>
# # **<center><span style="color:#00BFC4;"> EDA </span></center>**

# <a id="4.1"></a>
# ## <span style="color:#e76f51;"> Overview of Data </span>

# In[8]:


TARGET = ''
FEATURES = [col for col in df.columns if col != TARGET]
RANDOM_STATE = 12 


# In[9]:


df.iloc[:, :-1].describe().T.sort_values(by='std' , ascending = False)                     .style.background_gradient(cmap='GnBu')                     .bar(subset=["max"], color='#BB0000')                     .bar(subset=["mean",], color='green')


# <a href="#toc" role="button" aria-pressed="true" >⬆️Back to Table of Contents ⬆️</a>

# <a id="4.2"></a>
# ## <span style="color:#e76f51;">Continuos and Categorical Data Distribution </span>

# <div class="alert alert-block alert-info" style="font-size:14px; font-family:verdana; line-height: 1.7em;">
#     📌 &nbsp;<b><u>Observations in Features Distribution :</u></b><br>
#  
# * <i> Out of <b><u>9</u></b> features <b><u>all</u></b> features are continous.</i><br>
# </div>

# In[10]:


text_features = []
cat_features = [col for col in FEATURES if df[col].nunique() < 5 and col not in text_features ]
cont_features = [col for col in FEATURES if df[col].nunique() >= 5 and col not in text_features ]

print(f'\033[94mTotal number of features: {len(FEATURES)}')
print(f'\033[94mNumber of categorical features: {len(cat_features)}')
print(f'\033[94mNumber of continuos features: {len(cont_features)}')
print(f'\033[94mNumber of text features: {len(text_features)}')

labels=['Categorical', 'Continuos', "Text"]
values= [len(cat_features), len(cont_features), len(text_features)]
colors = ['#DE3163', '#58D68D']

fig = go.Figure(data=[go.Pie(
    labels=labels, 
    values=values, pull=[0.1, 0, 0 ],
    marker=dict(colors=colors, 
                line=dict(color='#000000', 
                          width=2))
)])
fig.show()


# <a href="#toc" role="button" aria-pressed="true" >⬆️Back to Table of Contents ⬆️</a>

# <a id="4.2"></a>
# ## <span style="color:#e76f51;"> Feature Distribution of Continous Features </span>

# ### <span style="color:#e76f51;">  Distribution of Int </span>

# In[11]:


df.select_dtypes('int').head(1)


# In[12]:


for col in df.select_dtypes('int'):
    df_Int = df.copy()
    fig = px.histogram(data_frame = df_Int, 
                       x=df[col],
                       color_discrete_sequence =  ['#58D68D','#DE3163'],
                       marginal="box",
                       nbins= 100,
                       template="plotly_white",
                       
                    )
    fig.update_layout(title = "Distribution of features"  , title_x = 0.5)
    fig.show()


# ### <span style="color:#e76f51;">  Distribution of float </span>

# In[13]:


df.select_dtypes('float').head(1)


# In[14]:


for col in df.select_dtypes('float'):
    df_Float = df.copy()
    fig = px.histogram(data_frame = df_Float, 
                       x=df[col],
                       color_discrete_sequence =  ['#58D68D','#DE3163'],
                       marginal="box",
                       nbins= 100,
                       template="plotly_white",
                       
                    )
    fig.update_layout(title = "Distribution of features"  , title_x = 0.5)
    fig.show()


# <a href="#toc" role="button" aria-pressed="true" >⬆️Back to Table of Contents ⬆️</a>

# <a id="4.6"></a>
# ## <span style="color:#e76f51;">  Map of correlations </span>

# In[15]:


plt.figure(figsize = (10,6))
mask = np.triu(np.ones_like(df.corr(), dtype=bool))
sns.heatmap(df.corr(), mask=mask, center=0, cmap='viridis', linewidths=1, annot=True, fmt='.2f', vmin=-1, vmax=1)
plt.title('Carte des corrélations')
plt.show()


# <a href="#toc" role="button" aria-pressed="true" >⬆️Back to Table of Contents ⬆️</a>

# <a id="5"></a>
# # **<center><span style="color:#00BFC4;">Data Pre-Processing  </span></center>**

# ### <span style="color:#e76f51;"> ACP </span>

# In[16]:


df_cor = df


# In[17]:


pipe = Pipeline([('scaler', StandardScaler()),
                 ('pca', PCA())])

pipe.fit(df_cor)

plt.figure(figsize = (10,6))
plt.bar(range(pipe[1].n_components_), pipe[1].explained_variance_ratio_)
plt.xlabel('Composantes principales')
plt.ylabel('Pourcentage de variance')
plt.xticks(range(pipe[1].n_components_))
plt.show()


# In[18]:


pipe[1].explained_variance_ratio_.cumsum()


# In[19]:


pipe2 = Pipeline([('scaler', StandardScaler()),
                 ('pca', PCA(n_components=7))])

df_transformed = pipe2.fit_transform(df_cor)


# <a href="#toc" role="button" aria-pressed="true" >⬆️Back to Table of Contents ⬆️</a>

# <a id="6"></a>
# # **<center><span style="color:#00BFC4;">Modeling </span></center>**

# <a id="6.1"></a>
# ## <span style="color:#e76f51;"> YellowBrick Cluster </span>

# In[20]:


kmeans = KMeans ()
plt.figure(figsize = (10,6))
visualizer = KElbowVisualizer(kmeans, k = (1,11))
visualizer.fit(df_transformed)
visualizer.show()


# In[21]:


model=KMeans(n_clusters=4)
model.fit(df_transformed)
predict = model.predict(df_transformed)


# <a href="#toc" role="button" aria-pressed="true" >⬆️Back to Table of Contents ⬆️</a>

# <a id="7"></a>
# # **<center><span style="color:#00BFC4;">Submission </span></center>**

# In[22]:


df = df.assign(Classe = predict)
df.head()


# In[23]:


df.Classe = df.Classe.map({0 : 'Classe A',1 : 'Classe B',2 : 'Classe C',3 : 'Classe D'})
df


# In[ ]:




