import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import plotly.graph_objs as go

pd.options.display.max_columns = 7
pd.options.display.max_rows = 7

# %% Data Analysis
data = pd.read_csv('D:/Masaüstüm/Projects/PythonProjects/Regression Types/K Nearest Neighbors/cancer_data.csv')

data.columns
"""
Index(['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst', 'Unnamed: 32'],
      dtype='object')
"""
data.drop(['id', 'Unnamed: 32'], axis = 1, inplace = True)

data['diagnosis'].value_counts()
"""
B    357 - benign
M    212 - malignant
"""

benign = data[data['diagnosis'] == 'B']
malignant = data[data['diagnosis'] == "M"]

for each,column in enumerate(benign.columns):
    if column == 'diagnosis':
        continue
    plt.figure(each)
    plt.scatter(benign.radius_mean, benign[column], color = 'green', label = 'benign')
    plt.scatter(malignant.radius_mean, malignant[column], color = 'red', label = 'malignant')
    plt.xlabel('radius_mean')
    plt.ylabel(column)
    plt.legend()
    plt.savefig("radius_mean and " + column + ".jpg")
    plt.show()
# they are in images folder, don't run if you dont want to.
# it generates more than 20 figures

data['diagnosis'] = [0 if each == 'B' else 1 for each in data['diagnosis']]
# since sklearn does not understand string(object), we convert to int.

x_n = data.drop(['diagnosis'], axis = 1)
y = data['diagnosis'].values.reshape(-1,1) 

x = (x_n - np.min(x_n)) / (np.max(x_n) - np.min(x_n)) # normalized

# %% train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.32, random_state = 13)

# %% KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 4) # neighbor counts
knn.fit(x_train,y_train)

knn.predict(x_test)

# %% evaluate model
print("{} neighbors score {}" .format(4, knn.score(x_test,y_test)))
# 4 neighbors score 0.9836065573770492

# %% Find Best K
list_k = []

for each in range(1,30):
    knn_n = KNeighborsClassifier(n_neighbors = each)
    knn_n.fit(x_train,y_train)
    list_k.append(knn_n.score(x_test,y_test))
    
plt.plot(range(1,30), list_k)
plt.xlabel("Neighbor Count")
plt.ylabel("Accuracy")
#plt.legend()
plt.show()    





