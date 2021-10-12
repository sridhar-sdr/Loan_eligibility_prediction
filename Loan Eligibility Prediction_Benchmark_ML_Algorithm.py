
# coding: utf-8

# # Task 10 : Benchmark Top ML Algorithms
# 
# This task tests your ability to use different ML algorithms when solving a specific problem.
# 

# ### Dataset
# Predict Loan Eligibility for Dream Housing Finance company
# 
# Dream Housing Finance company deals in all kinds of home loans. They have presence across all urban, semi urban and rural areas. Customer first applies for home loan and after that company validates the customer eligibility for loan.
# 
# Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have provided a dataset to identify the customers segments that are eligible for loan amount so that they can specifically target these customers.
# 
# Train: https://raw.githubusercontent.com/subashgandyer/datasets/main/loan_train.csv
# 
# Test: https://raw.githubusercontent.com/subashgandyer/datasets/main/loan_test.csv

# ## Task Requirements
# ### You can have the following Classification models built using different ML algorithms
# - Decision Tree
# - KNN
# - Logistic Regression
# - SVM
# - Random Forest
# - Any other algorithm of your choice

# ### Use GridSearchCV for finding the best model with the best hyperparameters

# - ### Build models
# - ### Create Parameter Grid
# - ### Run GridSearchCV
# - ### Choose the best model with the best hyperparameter
# - ### Give the best accuracy
# - ### Also, benchmark the best accuracy that you could get for every classification algorithm asked above

# #### Your final output will be something like this:
# - Best algorithm accuracy
# - Best hyperparameter accuracy for every algorithm
# 
# **Table 1 (Algorithm wise best model with best hyperparameter)**
# 
# Algorithm   |     Accuracy   |   Hyperparameters
# - DT
# - KNN
# - LR
# - SVM
# - RF
# - anyother
# 
# **Table 2 (Best overall)**
# 
# Algorithm    |   Accuracy    |   Hyperparameters
# 
# 

# ### Submission
# - Submit Notebook containing all saved ran code with outputs
# - Document with the above two tables

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


import seaborn as sns
sns.set_theme(style="darkgrid")
ax = sns.countplot(x="Loan_Status", data=data)


# Data is imbalanced

# In[5]:


data=pd.read_csv('https://raw.githubusercontent.com/subashgandyer/datasets/main/loan_train.csv')
data


# checking missing values

# In[6]:


data.isna().sum()


# In[7]:


data['Gender']= data['Gender'].fillna('U')
data['Married']= data['Married'].fillna('U')
data['Self_Employed']= data['Self_Employed'].fillna('U')


# In[8]:


data.isna().sum()


# In[9]:



from numpy import NaN
data[['LoanAmount','Loan_Amount_Term','Credit_History']] = data[['LoanAmount','Loan_Amount_Term','Credit_History']].replace(0, NaN)


# In[10]:


data.fillna(data.mean(), inplace=True)
data


# In[11]:


data.info()


# In[12]:


data.Dependents.value_counts()


# # Handling Categorical Variable

# In[13]:


from sklearn.preprocessing import LabelEncoder
labelencoder_X=LabelEncoder()
xm=data.apply(LabelEncoder().fit_transform)
xm


# In[14]:


X=xm.drop(['Loan_Status'], axis=1)
X


# In[15]:


y_new=xm.iloc[:,12]
y_new


# In[16]:


test=pd.read_csv('https://raw.githubusercontent.com/subashgandyer/datasets/main/loan_test.csv')


# In[17]:


test.isna().sum()


# In[18]:


test['Gender']= test['Gender'].fillna('U')
test['Self_Employed']= test['Self_Employed'].fillna('U')


# In[19]:


test.isna().sum()


# In[20]:


from numpy import NaN
test[['LoanAmount','Loan_Amount_Term','Credit_History']] = test[['LoanAmount','Loan_Amount_Term','Credit_History']].replace(0, NaN)


# In[21]:


test.fillna(test.mean(), inplace=True)
test.isna().sum()


# In[22]:


test.Dependents.value_counts()


# In[23]:


from sklearn.preprocessing import LabelEncoder
labelencoder_X=LabelEncoder()
xm_new=test.apply(LabelEncoder().fit_transform)
xm_new


# In[24]:


X.columns


# In[25]:


X_train_new=X[['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area']]
X_train_new

y_train_new=xm.iloc[:,12]
y_train_new


# In[26]:


X_train_new


# In[27]:


y_train_new


# In[28]:


X_test_new= xm_new
X_test_new


# In[29]:


from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y_new, test_size = 0.30, random_state=5)


# In[30]:


n = 247
 
# Dropping last n rows using drop
y_new.drop(y_new.tail(n).index,
        inplace = True)
 
# Printing dataframe
print(y_new)


# In[31]:



print(X_train_new.shape)
print(X_test_new.shape)
print(y_train_new.shape)
print(y_new.shape)


# In[32]:


y_test_new= y_new
y_test_new


# # Model Building

# In[33]:


from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier()


# In[34]:


knn_classifier.fit(X_train_new, y_train_new)


# In[35]:


knn_predictions = knn_classifier.predict(X_test_new)


# In[37]:


print(knn_classifier.score(X_test_new, y_test_new))
print(knn_classifier.score(X_train_new, y_train_new))


# # knn_classifier

# In[38]:


from sklearn.model_selection import GridSearchCV


# In[39]:


grid_params= {'n_neighbors':[3,5,11,19],'weights':['uniform','distance'],'metric':['euclidean','manhattan']
             }
            


# In[40]:


gridsearch= GridSearchCV(knn_classifier,grid_params, verbose=1,cv=3,n_jobs=-1)


# In[41]:


gs_results=gridsearch.fit(X_train_new, y_train_new)


# In[42]:


gs_results.best_score_


# In[43]:


gs_results.best_estimator_


# In[44]:


gs_results.best_params_


# # Random Forest With GridsearchCv

# In[45]:


from sklearn.ensemble import RandomForestClassifier
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)


# In[46]:


rf_results=grid_search.fit(X_train_new, y_train_new)


# In[58]:


rf_results.best_score_


# In[47]:


rf_results.best_params_


# # Decision Tree with GridSearchCv

# In[48]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}
grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, verbose=1, cv=3)
dt=grid_search_cv.fit(X_train_new, y_train_new)


# In[49]:


grid_search_cv.best_params_


# In[50]:


grid_search_cv.best_score_


# # Logistic Regression

# In[51]:


from sklearn.linear_model import LogisticRegression
import numpy as np


model=LogisticRegression()


# In[52]:


from sklearn.model_selection import RepeatedStratifiedKFold
# Create grid search object
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]
# define grid search
grid = dict(solver=solvers,penalty=penalty,C=c_values)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
lg = grid_search.fit(X_train_new, y_train_new)


# In[54]:


lg.best_score_


# In[53]:


lg.best_params_


# # svm

# In[ ]:


from sklearn.svm import SVC  
from sklearn.metrics import classification_report, confusion_matrix  
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from sklearn.model_selection import GridSearchCV
param_grid = { 'C':[0.1,1,100,1000],'kernel':['rbf','poly','sigmoid','linear'],'degree':[1,2,3,4,5,6],'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
grid = GridSearchCV(SVC(),param_grid)


# In[ ]:


grid.fit(X_train_new,y_train_new)


# In[59]:


grid.best_score_


# # Naivebayes

# In[55]:


import numpy as np
param_grid_nb = {
    'var_smoothing': np.logspace(0,-9, num=100)
}


# In[56]:


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
nbModel_grid = GridSearchCV(estimator=GaussianNB(), param_grid=param_grid_nb, verbose=1, cv=10, n_jobs=-1)
nbModel_grid.fit(X_train_new, y_train_new)
print(nbModel_grid.best_estimator_)
...
#Fitting 10 folds for each of 100 candidates, totalling 1000 fits
GaussianNB(priors=None, var_smoothing=1.0)


# In[57]:


print(nbModel_grid.best_score_)
print(nbModel_grid.best_params_)


# # AdaBoost Classifier

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
adb = AdaBoostClassifier(DecisionTreeClassifier(min_samples_split=10,max_depth=4),n_estimators=10,learning_rate=0.6)
adb.fit(X_train_new, y_train_new)
print("score on test: " + str(adb.score(X_test_new, y_test_new)))
print("score on train: "+ str(adb.score(X_train_new, y_train_new)))


# # BaggingClassifier

# In[61]:


from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
# max_samples: maximum size 0.5=50% of each sample taken from the full dataset
# max_features: maximum of features 1=100% taken here all 10K 
# n_estimators: number of decision trees 
bg=BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,max_features=1.0,n_estimators=10)
bg.fit(X_train_new, y_train_new)
print("score on test: " + str(bg.score(X_test_new, y_test_new)))
print("score on train: "+ str(bg.score(X_train_new, y_train_new)))


# # Voting Classifier

# In[62]:


from sklearn.ensemble import VotingClassifier
# 1) naive bias = mnb
# 2) logistic regression =lr
# 3) random forest =rf
# 4) support vector machine = svm
evc=VotingClassifier(estimators=[('gs_results',gs_results),('lg',lg),('rf_results',rf_results),('dt',dt),('bg',bg),('adb',adb)],voting='hard')
evc.fit(X_train_new, y_train_new)
print("score on test: " + str(evc.score(X_test_new, y_test_new)))
print("score on train: "+ str(evc.score(X_train_new, y_train_new)))

