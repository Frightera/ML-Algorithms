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

# %% ML Model
from sklearn import svm

n_svm = svm.SVC(kernel='rbf')
n_svm.fit(x_train,y_train)

y_hat = n_svm.predict(x_test)

# %% Evaluate
from sklearn.metrics import classification_report, confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Confusion matrix is a performance measurement for machine learning classification problem 
    where output can be two or more classes. It is a table with 4 different 
    combinations of predicted and actual values.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cnf_matrix = confusion_matrix(y_test, y_hat, labels=[0,1])
np.set_printoptions(precision=2)

print (classification_report(y_test, y_hat))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= True,  title='Confusion matrix')    

# %% Scikit Accuracy Scores

from sklearn.metrics import jaccard_score
jaccard_score(y_test, y_hat)
# 0.9508196721311475

y_hat = y_hat.reshape(-1,1)

from sklearn.metrics import f1_score
f1_score(y_test, y_hat, average='weighted') 
# 0.9835708624725525









