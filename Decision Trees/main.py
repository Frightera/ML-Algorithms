import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go

pd.options.display.max_columns = 6
pd.options.display.max_rows = 5

data = pd.read_csv("D:\Masaüstüm\Projects\PythonProjects\Regression Types\Decision Trees\drug200.csv")

# %% Check Data
data.head(-5)
"""
     Age Sex    BP Cholesterol  Na_to_K   Drug
0     23   F  HIGH        HIGH   25.355  drugY
1     47   M   LOW        HIGH   13.093  drugC
..   ...  ..   ...         ...      ...    ...
193   72   M   LOW        HIGH    6.769  drugC
194   46   F  HIGH        HIGH   34.686  drugY
"""

data.columns
"""
data.columns
Out[7]: Index(['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K', 'Drug'], dtype='object')
"""

data.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 200 entries, 0 to 199
Data columns (total 6 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   Age          200 non-null    int64  
 1   Sex          200 non-null    object 
 2   BP           200 non-null    object 
 3   Cholesterol  200 non-null    object 
 4   Na_to_K      200 non-null    float64
 5   Drug         200 non-null    object 
dtypes: float64(1), int64(1), object(4)
memory usage: 9.5+ KB
"""
# Sklearn Trees can not handle categorical variables, therefore we will convert
# "sex" to numerical variables.

X = data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]

X['Sex'] = [1 if each == 'M' else 0 for each in X['Sex']]
X['BP'] = [3 if each == 'HIGH' else 2 if each == 'NORMAL' else 1 for each in X['BP']]
X['Cholesterol'] = [1 if each == 'HIGH' else 0 for each in X['Cholesterol']]
"""
X.head()
Out[128]: 
   Age  Sex  BP  Cholesterol  Na_to_K
0   23    0   3            1   25.355
1   47    1   1            1   13.093
2   47    1   1            1   10.114
3   28    0   2            1    7.798
4   61    0   1            1   18.043
"""

X[['Na_to_K', 'Age']].astype(float)
y = data['Drug']
# Instead of list comph. we could have used labelencoder.
X.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 200 entries, 0 to 199
Data columns (total 5 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   Age          200 non-null    int64  
 1   Sex          200 non-null    int64  
 2   BP           200 non-null    int64  
 3   Cholesterol  200 non-null    int64  
 4   Na_to_K      200 non-null    float64
dtypes: float64(1), int64(4)
"""

# %% Set up Decision Tree
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.32, random_state = 3)

if X_train.shape == y_train.shape: # ensure that their size are matched.
    print("Sizes matched!")

# %% Modeling
from sklearn.tree import DecisionTreeClassifier

Tree = DecisionTreeClassifier(criterion="entropy", max_depth = 5)
Tree.fit(X_train, y_train)

y_hat = Tree.predict(X_test)

# %% Evaluation
from sklearn import metrics
#%matplotlib inline
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, y_hat))
# DecisionTrees's Accuracy:  0.984375

from six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree

dot_data = StringIO()
filename = "drugtree.png"
featureNames = data.columns[0:5]
targetNames = data["Drug"].unique().tolist()
out=tree.export_graphviz(Tree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_train), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')

