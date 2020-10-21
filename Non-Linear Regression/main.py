import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
sns.set(font_scale=1)  # you may change it for re-scaling.
#sns.set_theme(style="whitegrid")

pd.options.display.max_columns = 5
pd.options.display.max_rows = 13

data = pd.read_csv("D:\Masaüstüm\Projects\PythonProjects\Regression Types\Linear Regression\FuelConsumption.csv")

data.head(-5)
"""
0          2014  ACURA  ...                       33          196
1          2014  ACURA  ...                       29          221
        ...    ...  ...                      ...          ...
1060       2014  VOLVO  ...                       25          264
1061       2014  VOLVO  ...                       25          258
"""

data.isna().sum()
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
"""



#%%
def detect_outliers(data,features):
    outlier_indices = []
    for n in features:
       Q1 = np.percentile(data[n],25)
       Q3 = np.percentile(data[n],75)
       IQR = Q3-Q1
       outlier_step = IQR * 1.5
       outlier_list_col = data[(data[n]< Q1 - outlier_step) | (data[n]> Q3 + outlier_step)].index
       outlier_indices.extend(outlier_list_col) #store indices
    
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v>2)
    
    return multiple_outliers
# %% 
n_data = data[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

n_normalized = (n_data - np.min(n_data)) / (np.max(n_data) - np.min(n_data)).values
outliers = n_normalized.loc[detect_outliers(n_normalized, n_normalized.columns)]
n_normalized = n_normalized.drop(detect_outliers(n_normalized,n_normalized.columns),axis = 0).reset_index(drop = True)

outliers.head()
"""
     ENGINESIZE  CYLINDERS  ...  FUELCONSUMPTION_COMB  CO2EMISSIONS
182    0.702703   0.555556  ...              0.824645      0.647368
214    0.581081   0.555556  ...              0.796209      0.621053
216    0.581081   0.555556  ...              0.796209      0.621053
218    0.581081   0.555556  ...              0.848341      0.668421
220    0.581081   0.555556  ...              0.796209      0.621053
"""

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

# Boxplot
ax = sns.boxplot(data = n_data, orient="h", palette="Set3")
bx = sns.boxplot(data = n_normalized, orient="h", palette="Set3") # way better

# %% train test split
from sklearn.model_selection import train_test_split

n_train, n_test = train_test_split(n_data,test_size = 0.27, random_state = 13)
# Try to change n_data to n_normalized and see the difference
#These are random numbers(test size & state)
# You can change them to see the difference

# %% Model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


for each, column in enumerate(n_train.columns):
    if column == "CO2EMISSIONS":
        break
    train_x = n_train[column].values.reshape(-1,1)
    train_y = n_train['CO2EMISSIONS'].values.reshape(-1,1)

    test_x = n_test[column].values.reshape(-1,1)
    test_y = n_test['CO2EMISSIONS'].values.reshape(-1,1)

    poly = PolynomialFeatures(degree = 2)
    train_x_poly = poly.fit_transform(train_x) # we need to convert
    """
    fit_transform() takes our x values, 
    and output a list of our data raised 
    from power of 0 to power of 2
    """
    # Now, we can use LR to solve it.
    knb = LinearRegression()
    train_y_ = knb.fit(train_x_poly, train_y)
    
    # knb.coef_
    # knb.intercept_

    # %% Visualize Model
    #plt.scatter(n_train[each], n_train.CO2EMISSIONS,  color='blue')
    plt.figure(each)
    sns.scatterplot(data=n_train, x = column, y = "CO2EMISSIONS")
    XX = np.arange(0.0, 10.0, 0.1)
    yy = knb.intercept_[0]+ knb.coef_[0][1]*XX+ knb.coef_[0][2]*np.power(XX, 2)
    sns.lineplot(x=XX, y=yy)
    #plt.plot(XX, yy, '-r' )
    plt.xlabel(column)
    plt.ylabel("Emission")
    plt.show()
    """ Created a new figure for to display plots one by one. """
    
    # You can see all models, most of them wont suit.
    
    # %% Evaluate Model
    
    from sklearn.metrics import r2_score
    
    test_x_poly = poly.fit_transform(test_x)
    test_y_ = knb.predict(test_x_poly)
    
    print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
    print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
    print("R2-score: %.2f" % r2_score(test_y_ , test_y) )

    
    
    
