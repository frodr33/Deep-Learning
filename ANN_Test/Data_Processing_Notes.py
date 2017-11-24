# Data Preprocessing

# Importing the libraries
import numpy as np # Most of mathematics: Arrays
import matplotlib.pyplot as plt # plot nice charts
import pandas as pd # import and manage data sets

'''
Importing the dataset
Where X is the independent variable array
and Y is the Depedent variable array
'''

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values  # extract 3 first columns
y = dataset.iloc[:, 3].values # 3 or -1 does the same thing

'''
What about missing data?
In Excel Could remove lines that are missing..
More commonly, if there is missing data, replace
missing data with mean of values in that column
'''
# take care of missing data
from sklearn.preprocessing import Imputer
# Use Imputer to fill in nans with mean straegy along columns
# NOTE: CTRL + i brings up API Specification
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)

# 2nd and 3rd columns have missing data
# upperbound in brackets are excluded in python
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])  # replaces data


####################################################
'''
Preprocessing Step 4:
Categorical Variables contain different categories. 
For example, the Country column has 3 categories, i.e
France, Spain, and Germany
Need to Encode text into numbers

Dummy Encoding:
Change the one column that results from categorical
encoding two 3 columns because you dont want 1 to be mapped
to Germany and 2 to France because then you can say that 
France is greater than Germany which makes no sense. Instead, 
with 3 columns you will have 1's or 0's

Dont have to use Dummy Encoding for Dependent variable
because the machine will know that it is indeed the Depednent 
variable
'''

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_Country = LabelEncoder()
X[:, 0] = labelencoder_Country.fit_transform(X[:, 0]) # want first column
oneHotEncoder = OneHotEncoder(categorical_features = [0])
X = oneHotEncoder.fit_transform(X).toarray()
labelencoder_DV = LabelEncoder()
y = labelencoder_DV.fit_transform(y)

########################################################

'''
Preprocessing Step 5
Need to split data set into training set and test set
Build machine learning algorithm using training set 
but need to performance on test set
'''

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#########################################################

'''
Preprocssing Step 6: Feature Scaling
Variables may not be on same scale, for example, Age goes from 7-50 and
salary goes from 40k to 90k, this will cause some issues in ML model
This is because a lot of models use Euclidean Distance...
Since Salary has a wider range than the age, the Euclidean distance
will be dominated by the salary

We transform values from -1 to +1
There are two types of feature scaling
1). Standardisation: 
    Xstand = (x - mean(X)) / standarddeviation(X)
2). Normalisation:
    Xnorm = (x - min(X)) / (max(X) - min(X))

Need to scale dummy variables, sometimes...depends on the context
we do here (by simply not specifying ranges in X)

NOTE: Do not need to imply feature scaling to Y vales because
the DV is categorical and only takes 1 or 0. Will need to apply feature scaling
to DV if there are many possible values
'''

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) # dont need fit fit bc train has it












