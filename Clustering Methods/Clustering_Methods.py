#!/usr/bin/env python
# coding: utf-8

# # Clustering Methods

# ## The Data
# 
# This time a wine quality dataset is being used. The data set contains various chemical properties of wine, such as acidity, sugar, pH, and alcohol.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from colorsetup import colors, palette
sns.set_palette(palette)

print('Libraries are imported :)')


# ## Data Analysis
# 
# * Let's examine which ones are categorical or numerical etc.

# In[2]:


data = pd.read_csv('Wine_Quality_Data.csv')

data.head(10)


# In[3]:


data.shape


# * "The implementation of K-means in Scikit-learn is designed only to work with continuous data (even though it is sometimes used with categorical or boolean types)." 
# * Every feature except quality and color are continuous. ( Cont. means can take any value ) 

# In[4]:


data.dtypes


# The number of entries for each wine color.

# In[5]:


(data['color'].value_counts())


# In[6]:


data['quality'].value_counts().sort_index(ascending = False)


# The distribution of quality values. We have 7 different qualities.

# In[7]:


sns.set_theme()
plt.figure(figsize= (13,9))
sns.histplot(x = 'quality', hue = 'color', data = data, bins = len(data['quality'].value_counts()), element = 'step')
plt.xlabel('Quality')
plt.ylabel('Counts')
plt.show()


# #### Examining the correlation and skew of the relevant variables
# 

# In[8]:


float_columns = [x for x in data.columns if x not in ['color', 'quality']]

# The correlation matrix
corr_mat = data[float_columns].corr()

# Strip out the diagonal values for the next step
for x in range(len(float_columns)):
    corr_mat.iloc[x,x] = 0.0
    
corr_mat


# In[9]:


# Pairwise maximal correlations
corr_mat.abs().idxmax()


# And an examination of the skew values in anticipation of transformations.

# In[10]:


skew_columns = (data[float_columns]
                .skew()
                .sort_values(ascending=False))

skew_columns = skew_columns.loc[skew_columns > 0.75]
skew_columns


# In[11]:


# Perform log transform on skewed columns
for col in skew_columns.index.tolist():
    data[col] = np.log1p(data[col])


# Perform feature scaling.

# In[12]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
data[float_columns] = sc.fit_transform(data[float_columns])

data.head(-5)


# In[33]:


# Alternatively

sns.set_theme(style="white")

# Compute the correlation matrix
corr = data.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(22, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, annot = True, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# The pairplot of the transformed and scaled features.

# In[13]:


sns.set_theme()
sns.pairplot(data = data, 
             hue='color', 
             hue_order=['white', 'red'])


# ### Fit a K-means clustering model with three clusters

# In[14]:


from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, random_state=13)
km = km.fit(data[float_columns])

data['kmeans_labels'] = km.predict(data[float_columns])


# In[15]:


data['kmeans_labels'].head(-5)


# In[16]:


(data[['color','kmeans_labels']]
 .groupby(['kmeans_labels','color'])
 .size()
 .to_frame()
 .rename(columns={0:'number'}))


# ### Finding best K
# 
# * Fit K-Means models with cluster values ranging from 1 to 20.
# * For each model, store the number of clusters and the inertia value. 

# In[17]:


distortions = []
K = range(1,22)
for k in K:
    kmean = KMeans(n_clusters=k, random_state= 13, n_init = 50, max_iter = 500)
    kmean.fit(data[float_columns])
    distortions.append(kmean.inertia_)

plt.figure(figsize=(13,5))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method')
plt.show()


# * Were you able to spot the best k?

# ### Fitting an agglomerative clustering model with three clusters.
# 

# In[18]:


from sklearn.cluster import AgglomerativeClustering
ag = AgglomerativeClustering(n_clusters=3, linkage='ward', compute_full_tree=True)
ag = ag.fit(data[float_columns])
data['aggCL'] = ag.fit_predict(data[float_columns])
data['aggCL'].head(-5)


# Note that cluster assignment is arbitrary, the respective primary cluster numbers for red and white may not be identical to the ones below and also may not be the same for both K-means and agglomerative clustering.

# In[19]:


# First, for Agglomerative Clustering:
(data[['color','aggCL','kmeans_labels']]
 .groupby(['color','aggCL'])
 .size()
 .to_frame()
 .rename(columns={0:'number'}))


# In[20]:


# Comparing with KMeans results:
(data[['color','aggCL','kmeans_labels']]
 .groupby(['color','kmeans_labels'])
 .size()
 .to_frame()
 .rename(columns={0:'number'}))


# In[21]:


# Comparing results:
(data[['color','aggCL','kmeans_labels']]
 .groupby(['color','aggCL','kmeans_labels'])
 .size()
 .to_frame()
 .rename(columns={0:'number'}))


# * Plot of the dendrogram created from agglomerative clustering.

# In[25]:


from scipy.cluster import hierarchy

Z = hierarchy.linkage(ag.children_, method='ward')

fig, ax = plt.subplots(figsize=(22,9))

# Some color setup
red = colors[2]
blue = colors[0]

hierarchy.set_link_color_palette([red, 'gray'])

den = hierarchy.dendrogram(Z, orientation='top', 
                           p=30, truncate_mode='lastp',
                           show_leaf_counts=True, ax=ax,
                           above_threshold_color=blue)


# ### Clustering as a form of feature engineering
# 
# * Compare the average roc-auc scores for both models, the one using the KMeans cluster as a feature and the one that doesn't use it.
# 

# In[29]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

y = (data['quality'] > 6).astype(int)
X_with_kmeans = data.drop(['aggCL', 'color', 'quality'], axis=1)
X_without_kmeans = X_with_kmeans.drop('kmeans_labels', axis=1)
sss = StratifiedShuffleSplit(n_splits=13, random_state=13)

def get_avg_roc_13splits(estimator, X, y):
    roc_auc_list = []
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        estimator.fit(X_train, y_train)
        y_predicted = estimator.predict(X_test)
        y_scored = estimator.predict_proba(X_test)[:, 1]
        roc_auc_list.append(roc_auc_score(y_test, y_scored))
    return np.mean(roc_auc_list)

estimator = RandomForestClassifier()
roc_with_kmeans = get_avg_roc_13splits(estimator, X_with_kmeans, y)
roc_without_kmeans = get_avg_roc_13splits(estimator, X_without_kmeans, y)
print("Without kmeans cluster as input to Random Forest, roc-auc is --> \"{0}\"".format(roc_without_kmeans))
print("Using kmeans cluster as input to Random Forest, roc-auc is --> \"{0}\"".format(roc_with_kmeans))


# * Fit 13 **Logistic Regression** models and compute the average roc-auc-score

# In[30]:


import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression

X_basis = data[float_columns]
sss = StratifiedShuffleSplit(n_splits=13, random_state=13)

def create_kmeans_columns(n):
    km = KMeans(n_clusters=n)
    km.fit(X_basis)
    km_col = pd.Series(km.predict(X_basis))
    km_cols = pd.get_dummies(km_col, prefix='kmeans_cluster')
    return pd.concat([X_basis, km_cols], axis=1)

estimator = LogisticRegression()
ns = range(1, 21)
roc_auc_list = [get_avg_roc_13splits(estimator, create_kmeans_columns(n), y)
                for n in ns]

ax = plt.axes()
ax.plot(ns, roc_auc_list)
ax.set(
    xticklabels= ns,
    xlabel='Number of clusters as features',
    ylabel='Average ROC-AUC over 13 iterations',
    title='KMeans + LogisticRegression'
)
ax.grid(True)


# In[ ]:




