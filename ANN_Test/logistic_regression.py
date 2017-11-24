# Logistic Regression

# Data Preprocessing

# Importing the libraries
import numpy as np # Most of mathematics: Arrays
import matplotlib.pyplot as plt # plot nice charts
import pandas as pd # import and manage data sets

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:,4].values


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) # dont need fit fit bc train has it

# Fitting Logisitic Reogression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predcting the Test set results
y_pred = classifier.predict(X_test) # Predict y given X_test

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) # 64, 24 are correct predictions, 8 and 3 are incorret

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train # creates local variables

'''
Create Graph, where range on X axis [min(age) - 1 ... max(age) + 1]
              where range on Y axis [min(salary) -1 ... max(salary) + 1]
              Step = 0.01, means 0.01 resolution       
'''
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))


'''
Apply classifier on all pixel observation points. It looks at X1 and X2 of each pixel
and predicts whether it should be yes or no. If yes it colors pixel green
and if no it colors pixel red. Then it draws contour line between the 
red and green regions
'''
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))


plt.xlim(X1.min(), X1.max()) # Set x limits of current axis
plt.ylim(X2.min(), X2.max())

'''
Plot all the data points...somehow...review the syntax. High level
overview is that using an array created by y_set, the for each loop
iterates through and then plots the points in the X_set one by one while 
also coloring them green or red depending on if point is true or false
**i is always either 1 or 0...is this true? its how its used for listedcolormap
**j is index in y_set array

'''
for i, j in enumerate(np.unique(y_set)): # Iterate through all sorted unique elements in y_set array
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j) # red is 0 and green is 1, i is either 1 or 0
    
plt.title('Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

'''
How to visualization/Graph was made.........







'''
# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()












