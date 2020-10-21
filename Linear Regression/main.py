import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go

pd.options.display.max_columns = 5
pd.options.display.max_rows = 5

data = pd.read_csv("D:\Masaüstüm\Projects\PythonProjects\Regression Types\Linear Regression\FuelConsumption.csv")

# %% Take a look at the data 

data.head(10)

"""
    MODELYEAR   MAKE  ... FUELCONSUMPTION_COMB_MPG CO2EMISSIONS
0        2014  ACURA  ...                       33          196
1        2014  ACURA  ...                       29          221
..        ...    ...  ...                      ...          ...
8        2014  ACURA  ...                       24          267
9        2014  ACURA  ...                       31          212
"""

data.columns
"""
Index(['MODELYEAR', 'MAKE', 'MODEL', 'VEHICLECLASS', 'ENGINESIZE', 'CYLINDERS',
       'TRANSMISSION', 'FUELTYPE', 'FUELCONSUMPTION_CITY',
       'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB',
       'FUELCONSUMPTION_COMB_MPG', 'CO2EMISSIONS'],
      dtype='object')
"""

data.describe()
"""
       MODELYEAR   ENGINESIZE  ...  FUELCONSUMPTION_COMB_MPG  CO2EMISSIONS
count     1067.0  1067.000000  ...               1067.000000   1067.000000
mean      2014.0     3.346298  ...                 26.441425    256.228679
         ...          ...  ...                       ...           ...
75%       2014.0     4.300000  ...                 31.000000    294.000000
max       2014.0     8.400000  ...                 60.000000    488.000000
"""

data.isna().sum()
#Detect missing values.
"""
MODELYEAR                   0
MAKE                        0
MODEL                       0
VEHICLECLASS                0
ENGINESIZE                  0
CYLINDERS                   0
TRANSMISSION                0
FUELTYPE                    0
FUELCONSUMPTION_CITY        0
FUELCONSUMPTION_HWY         0
FUELCONSUMPTION_COMB        0
FUELCONSUMPTION_COMB_MPG    0
CO2EMISSIONS                0
dtype: int64
"""
# %%

n_data = data[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
n_normalized = (n_data - np.min(n_data)) / (np.max(n_data) - np.min(n_data)).values

for each in n_data:
    g = n_data[[each]]
    g.hist()
    plt.show()
    
# We can plot each of these vs to find a relation.
g = sns.PairGrid(n_normalized)
g.map(sns.regplot)

# Correlation
sns.heatmap(n_normalized.corr(), annot = True, fmt = ".2f")
plt.show()

# %% train test split
from sklearn.model_selection import train_test_split

n_train, n_test = train_test_split(n_normalized,test_size = 0.21, random_state = 42)

# %% Regression 
from sklearn.linear_model import LinearRegression
LinearReg = LinearRegression()

n_train_engine = np.asanyarray(n_train[['ENGINESIZE']])
n_train_emission = np.asanyarray(n_train[['CO2EMISSIONS']])

LinearReg.fit(n_train_engine, n_train_emission)
LinearReg.coef_
"""
LinearReg.coef_
Out[77]: array([[0.7577476]])
"""

LinearReg.intercept_
"""
LinearReg.intercept_
Out[78]: array([0.15048711])
"""

plt.scatter(n_train.ENGINESIZE, n_train.CO2EMISSIONS,  color='red')
plt.plot(n_train_engine, LinearReg.coef_[0][0]*n_train_engine + LinearReg.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")

# %% Evaluate
from sklearn.metrics import r2_score

n_test_engine = np.asanyarray(n_test[['ENGINESIZE']])
n_test_emission = np.asanyarray(n_test[['CO2EMISSIONS']])

n_test_predict = LinearReg.predict(n_test_engine)

print("Mean absolute error: %.2f" % np.mean(np.absolute(n_test_predict - n_test_engine)))
print("Residual sum of squares (MSE): %.2f" % np.mean((n_test_predict - n_test_engine) ** 2))
print("R2-score: %.2f" % r2_score(n_test_engine , n_test_predict) )

"""
Mean absolute error: 0.08
Residual sum of squares (MSE): 0.01
R2-score: 0.79
"""
