#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing libraries
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import VotingClassifier
from vecstack import stacking


# In[ ]:


# Importing dataset
dataset = pd.read_csv('diabetes.csv')


# In[ ]:


# Preview data
dataset.head()


# In[ ]:


# Dataset dimensions - (rows, columns)
dataset.shape


# In[ ]:


# Features data-type
dataset.info()


# In[ ]:


# Statistical summary
dataset.describe().T


# In[ ]:


dataset_new = dataset


# In[ ]:


#missing values
dataset_new[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = dataset_new[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.NaN) 


# In[ ]:


# Count of NaN
dataset_new.isnull().sum()


# In[ ]:


# Impute mean values
dataset_new["Glucose"].fillna(dataset_new["Glucose"].mean(), inplace = True)
dataset_new["BloodPressure"].fillna(dataset_new["BloodPressure"].mean(), inplace = True)
dataset_new["SkinThickness"].fillna(dataset_new["SkinThickness"].mean(), inplace = True)
dataset_new["Insulin"].fillna(dataset_new["Insulin"].mean(), inplace = True)
dataset_new["BMI"].fillna(dataset_new["BMI"].mean(), inplace = True)


# In[ ]:


# Selecting features - [Glucose, Insulin, BMI, Age]
X = dataset_new.iloc[:, [1, 4, 5, 7]].values
Y = dataset_new.iloc[:, 8].values


# In[ ]:


# Splitting X and Y
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42, stratify = dataset_new['Outcome'] )


# In[ ]:


# performing test validation split

validation_ratio = 0.20
test_ratio = 0.10
X_val, X_test, Y_val, Y_test = train_test_split(
    X_test, Y_test, test_size=test_ratio/(test_ratio + validation_ratio))
 


# In[ ]:


# Checking dimensions
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)


# In[ ]:


# Logistic Regression Algorithm
from sklearn.linear_model import LogisticRegression
logregression = LogisticRegression(random_state = 42)
logregression.fit(X_train, Y_train)


# In[ ]:


# K nearest neighbors Algorithm
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 24, metric = 'minkowski', p = 2)
knn.fit(X_train, Y_train)


# In[ ]:


# Support Vector Machine
from sklearn.svm import SVC
svm = SVC(kernel = 'linear', random_state = 42)
svm.fit(X_train, Y_train)


# In[ ]:


# Naive Bayes Algorithm
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, Y_train)


# In[ ]:


# Decision tree Algorithm
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(criterion = 'entropy', random_state = 42)
dtree.fit(X_train, Y_train)


# In[ ]:


# Random forest Algorithm
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 11, criterion = 'entropy', random_state = 42)
rf.fit(X_train, Y_train)


# In[ ]:


# Making predictions on test dataset
Y_pred_logregression = logregression.predict(X_test)
Y_pred_knn = knn.predict(X_test)
Y_pred_svm = svm.predict(X_test)
Y_pred_nb = nb.predict(X_test)
Y_pred_dtree = dtree.predict(X_test)
Y_pred_rf = rf.predict(X_test)


# In[ ]:


# Evaluating using accuracy_score metric
from sklearn.metrics import accuracy_score
accuracy_logregression = accuracy_score(Y_test, Y_pred_logregression)
accuracy_knn = accuracy_score(Y_test, Y_pred_knn)
accuracy_svm = accuracy_score(Y_test, Y_pred_svm)
accuracy_nb = accuracy_score(Y_test, Y_pred_nb)
accuracy_dtree = accuracy_score(Y_test, Y_pred_dtree)
accuracy_rf = accuracy_score(Y_test, Y_pred_rf)


# In[ ]:


# Accuracy on test set
print("Logistic Regression: " + str(accuracy_logregression * 100))
print("K Nearest neighbors: " + str(accuracy_knn * 100))
print("Support Vector Classifier: " + str(accuracy_svm * 100))
print("Naive Bayes: " + str(accuracy_nb * 100))
print("Decision tree: " + str(accuracy_dtree * 100))
print("Random Forest: " + str(accuracy_rf * 100))


# In[ ]:


# Confusion matrix
from sklearn.metrics import confusion_matrix
confusionMatrix = confusion_matrix(Y_test, Y_pred_knn)
confusionMatrix


# In[ ]:


# Classification report
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred_knn))


# In[ ]:


#stacking


all_models = [logregression, knn, svm, nb, dtree, rf]
s_train, s_test = stacking(all_models, X_train, X_test,
                           Y_train, regression=True, n_folds=4)
final_model = model_1
final_model = final_model.fit(s_train, Y_train)
pred_final = final_model.predict(X_test)
print(mean_squared_error(Y_test, pred_final))


# In[ ]:


#averaging

pred_final = (Y_pred_logregression+Y_pred_knn+Y_pred_svm+Y_pred_nb+Y_pred_dtree+Y_pred_rf)/6.0
print(mean_squared_error(Y_test, pred_final))


# In[ ]:


#max vote

from sklearn.metrics import log_loss

final_model = VotingClassifier(
    estimators=[('lr', logregression), ('knn', knn), ('rf', rf)], voting='hard')
 
# training all the model on the train dataset
final_model.fit(X_train, Y_train)
 
# predicting the output on the test dataset
pred_final = final_model.predict(X_test)
 
# printing log loss between actual and predicted value
print(log_loss(Y_test, pred_final))


# In[ ]:




