#!/usr/bin/env python
# coding: utf-8

# # Categorical Naive Bayes and Gaussian Naive Bayes for Weather & Breast Cancer Data Analysis
# 
# 

# ## Importing Dependencies

# In[1]:


import numpy as np
import pandas as pd
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn import datasets
import seaborn as sns


# ## Task 1

# ### Reading Dataset
# 
# - Dataset was first stored as a csv file manually from the given values in image format
# - Then imported as a dataframe

# In[3]:


df = pd.read_csv("weather.csv")


# In[4]:


df


# ### Implementation of Naive Bayes Posterior Calculation Class
# 
# For ease of calculation at later stages and because of personal interest implemented the Naive Bayes posterior Calculation class.
# 
# - Class can calculate prior, likelihood and posterior, and also perform complete calculation uptil posterior with a single function.
# - Class functionality was `verified by manual calculation`.

# In[187]:


class NaiveBayesPosterior():
    def __init__(self, df, features, target, lam = 0):
        self.features = features
        self.target = target
        self.df = df
        self.prior = 0
        self.likelihood = {}
        self.posterior = 0
        self.lam = lam
        
    def calculate_prior(self):
        if (self.lam == 0):
            self.prior = self.df.groupby(self.target).size().div(len(self.df))
        else:
            size = self.df.groupby(self.target).size() + self.lam
            total = self.df.groupby(self.target).size().sum() +                 self.lam * len(self.df.groupby(self.target).size().index)
            self.prior = size / total
        
    def calculate_likelihood(self):
        if (self.lam == 0):    
            for feature in self.features:
                self.likelihood[feature] = self.df.groupby([self.target,feature]).size().div(len(self.df)).div(self.prior)
        else:
            for feature in self.features:
                size = self.df.groupby([self.target,feature]).size() + self.lam
                count = len(self.df) + self.lam*(len(self.df.groupby(feature).size().index))
                self.likelihood[feature] = (size / count) / self.prior
        
            
    def calculate_posterior(self, values):
        self.post_calc = {}
        self.post_calc['Yes'] = self.likelihood[values[0][0]]['Yes'][values[0][1]] *             self.likelihood[values[1][0]]['Yes'][values[1][1]] * self.prior['Yes']
        
        self.post_calc['No'] = self.likelihood[values[0][0]]['No'][values[0][1]] *             self.likelihood[values[1][0]]['No'][values[1][1]] * self.prior['No']
        
        self.denominator_factor = self.post_calc['Yes'] + self.post_calc['No']
        
        self.posterior = self.post_calc[values[-1]] / self.denominator_factor
        
        return self.posterior
    
    def complete_calculation(self, values):
        self.calculate_prior()
        self.calculate_likelihood()
        self.calculate_posterior(values)


# In[208]:


# Creating the object to be used later
# Passing second attribute of feature names and last one of target column name
customNB = NaiveBayesPosterior(df, df.columns[:-1],df.columns[-1])


# ### Part 1 - Verification of the filled table
# 
# - The required table was filled manually by looking at the dataset
# - The probabilities calculated manually can be verified using this automated solution

# In[209]:


customNB.df.head()


# In[210]:


# Calculating prior and likelihood
customNB.calculate_prior()
customNB.calculate_likelihood()


# The table values can be verified by comparing `prior` and `likelihood` values below

# In[212]:


customNB.prior


# In[115]:


customNB.likelihood


# In[88]:


customNB.likelihood


# ### Part 2 - Posterior Calculation
# 
# - Posterior was first calculated manually and then using custom Naive Bayes class implemented above
# - Results obtained were `verified` and turned out to be `consistent`

# In[196]:


# The posterior calculation function takes input of a nested list of the following format
values = [['weather','Sunny'],['temperature','Hot'], 'Yes']


# In[12]:


customNB.likelihood


# In[13]:


# Calculating posterior of Play='Yes' contioned on weather = "Sunny" and Temperature = "Hot"
customNB.complete_calculation(values)


# The required **Posterior** is:

# In[206]:


print("The value of posterior is {}".format(customNB.posterior))


# ### Part 3 - Posterior Calculation with Smoothing
# 
# Implementation for calculation of posterior using given values of `lambda` was also added to the custom Naive Bayes implementation

# In[199]:


# Last parameter is value of lambda
customNB = NaiveBayesPosterior(df, df.columns[:-1],df.columns[-1],1)


# In[200]:


# Calculating prior and likelihood
customNB.calculate_prior()
customNB.calculate_likelihood()


# In[201]:


values = [['weather','Sunny'],['temperature','Hot'], 'Yes']


# In[202]:


customNB.complete_calculation(values)


# The required value of `posterior` with `lambda=1` is:

# In[207]:


print("The value of posterior is {}".format(customNB.posterior))


# ### Part 4

# #### Calculations of Posterior with smoothing parameters

# In[246]:


customNB = NaiveBayesPosterior(df, df.columns[:-1],df.columns[-1],0)
customNB.calculate_prior()
customNB.calculate_likelihood()
customNB.complete_calculation(values)
print("The value of posterior for lambda= 0 is {}".format(customNB.posterior))


# In[247]:


customNB = NaiveBayesPosterior(df, df.columns[:-1],df.columns[-1],1)
customNB.calculate_prior()
customNB.calculate_likelihood()
customNB.complete_calculation(values)
print("The value of posterior for lambda= 1 is {}".format(customNB.posterior))


# #### Encoding the Dataset with Ordinal Encoder

# In[248]:


enc = OrdinalEncoder()


# In[249]:


enc = enc.fit(df)


# In[250]:


enc.categories_


# In[251]:


df_encoded = pd.DataFrame(enc.transform(df))
df_encoded.columns = df.columns
df_encoded


# In[252]:


X = df_encoded.iloc[:,0:2]
y = df_encoded.iloc[:,-1]
y


# In[253]:


enc.categories_[2][0]


# In[283]:


encoded = enc.transform(df)
X = encoded[:,0:2]
y = encoded[:,-1]
encoded


# In[290]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


# In[291]:


X_train


# #### Building the classifier

# First calculating with alpha (lambda) = 0

# In[292]:


cnb = CategoricalNB(alpha=0)


# In[293]:


cnb


# In[294]:


cnb = cnb.fit(X_train,y_train)


# In[295]:


predictions = cnb.predict(X_train)
predictions


# In[298]:


y_train_pred = [enc.categories_[2][int(y)] for y in predictions]
y_train_orig = [enc.categories_[2][int(y)] for y in y_train]
y_train_pred


# In[304]:


print("The accuracy value when lambda = 0 on train set is {}".format(accuracy_score(y_train_orig,y_train_pred)))


# In[305]:


predictions = cnb.predict(X_test)
predictions


# In[306]:


y_test_pred = [enc.categories_[2][int(y)] for y in predictions]
y_test_orig = [enc.categories_[2][int(y)] for y in y_test]
y_test_pred


# In[307]:


print("The accuracy value when lambda = 0 on train set is {}".format(accuracy_score(y_test_orig,y_test_pred)))


# Building classifier with alpha (lambda) = 1

# In[308]:


cnb = CategoricalNB(alpha=1)


# In[311]:


cnb = cnb.fit(X_train,y_train)


# In[312]:


predictions = cnb.predict(X_train)
predictions


# In[313]:


y_train_pred = [enc.categories_[2][int(y)] for y in predictions]
y_train_orig = [enc.categories_[2][int(y)] for y in y_train]
y_train_pred


# In[314]:


print("The accuracy value when lambda = 0 on train set is {}".format(accuracy_score(y_train_orig,y_train_pred)))


# In[315]:


predictions = cnb.predict(X_test)
predictions


# In[316]:


y_test_pred = [enc.categories_[2][int(y)] for y in predictions]
y_test_orig = [enc.categories_[2][int(y)] for y in y_test]
y_test_pred


# In[317]:


print("The accuracy value when lambda = 0 on train set is {}".format(accuracy_score(y_test_orig,y_test_pred)))


# ## Task 2

# ### Reading Data

# In[32]:


df = pd.read_csv("2_analcatdata_broadway.csv", sep=";")
df.head()


# ### Data Encoding
# 
# Initially, it looked like that encoding was not required as data appears to be in required format, but the classifier generates an error when negative values are passed. Therefore, it has to be processed.

# In[33]:


enc = OrdinalEncoder()


# In[34]:


enc.fit(df)


# In[35]:


enc.categories_


# In[36]:


df_t = enc.transform(df)
df_t = pd.DataFrame(df_t)
df_t.columns = df.columns
df_t


# ### Seperating and Splitting
# 
# First seperating the dataset into features and labels and then splitting them into 20% test and 80% train datasets

# In[37]:


# Separating features and labels
X = df_t.loc[:,df_t.columns!="label"]
y = df_t["label"]


# In[38]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


# In[39]:


X_test


# ### Building the Classifier and doing a test run

# In[40]:


cnb = CategoricalNB()


# In[41]:


cnb.fit(X_train, y_train)


# In[42]:


# Predicting on train data
predictions = cnb.predict(X_train)
predictions


# In[43]:


# Transforming back to original values (Doing manual, selective inverse transform from encoder)
y_pred = [enc.categories_[-1][int(y)] for y in predictions]
y_true = [enc.categories_[-1][int(y)] for y in y_train]


# In[45]:


X_train


# In[46]:


predicion_array = {'train_acc':[], 'test_acc':[], 'train_error':[], 'test_error':[]}
for i in range(50+1):
    cnb = CategoricalNB(alpha=i)
    cnb.fit(X_train, y_train)
    train_pred = cnb.predict(X_train)
    test_pred = cnb.predict(X_test)
    y_train_pred = [enc.categories_[-1][int(y)] for y in train_pred]
    y_test_pred = [enc.categories_[-1][int(y)] for y in test_pred]
    y_train_true = [enc.categories_[-1][int(y)] for y in y_train]
    y_test_true = [enc.categories_[-1][int(y)] for y in y_test]
    train_accuracy = accuracy_score(y_train_true, y_train_pred)
    test_accuracy = accuracy_score(y_test_true, y_test_pred)
    predicion_array['train_acc'].append(train_accuracy)
    predicion_array['test_acc'].append(test_accuracy)
    predicion_array['train_error'].append(1 - train_accuracy)
    predicion_array['test_error'].append(1 - test_accuracy)


# In[47]:


predicion_array


# In[60]:


x_axis = [x for x in range(51)]
plt.figure(figsize=(16,10), dpi= 80)
no_of_features = list(range(1,len(X.columns)+1))
plt.plot(x_axis, predicion_array['train_acc'], color='tab:red', label='Train Accuracy')
plt.plot(x_axis, predicion_array['test_acc'], color="steelblue", label='Test Accuracy')
plt.ylabel("Accuracy", fontsize=12)
plt.xlabel("Alpha", fontsize=12)
plt.yticks(fontsize=12, alpha=.9)
plt.title("Train and Test Accuracy for varying Alpha (Categorical NB)", fontsize=16)
plt.axvline(1,ls='--', color='black')
plt.legend()
plt.show()


# In[58]:


x_axis = [x for x in range(51)]
plt.figure(figsize=(16,10), dpi= 80)
no_of_features = list(range(1,len(X.columns)+1))
plt.plot(x_axis, predicion_array['train_error'], color='tab:green', label='Train Error')
plt.plot(x_axis, predicion_array['test_error'], color="tab:red", label='Test Error')
plt.ylabel("Error", fontsize=12)
plt.xlabel("Alpha", fontsize=12)
plt.yticks(fontsize=12, alpha=.9)
plt.title("Train and Test Error for varying Alpha (Categorical NB)", fontsize=16)
plt.axvline(1,ls='--', color='black')
plt.legend()
plt.show()


# **Observations**
# 
# Looking at the results, increasing the alpha starts an immediate decline in train accurancy, but it still does not improve the test accuracy and that too starts to drop as the alpha increases. For this dataset, keeping small alpha value would be more favorable.

# ### Best Value of Smoothing Parameter - Lambda (alpha)
# 
# For the given dataset, it can be observed from both the `accuracy` and `error` plots, that the best value of smoothing parameter is `1` as it is giving minimum error, which is shown using black dotted line on the graphs.

# ## Task 3 - Gaussian Naïve Bayes

# ### Importing Dataset

# In[111]:


breast_cancer = datasets.load_breast_cancer()


# In[117]:


breast_cancer['feature_names']


# In[135]:


c_df = pd.DataFrame(data = breast_cancer['data'], 
                    columns = breast_cancer['feature_names'])
c_df['target'] = breast_cancer['target']


# In[136]:


c_df.head()


# ### Visualizing the Dataset Features Correlation

# In[140]:


# Looking at correlation among the dataset features
corr = c_df.corr()
corr.shape


# **Do not run again, takes times to generate the visualization**

# In[141]:


plt.figure(figsize=(20,20))
sns.heatmap(corr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':15}, cmap='Greens')
plt.show()


# In[145]:


c_df.columns


# In[147]:


breast_cancer.target_names


# In[148]:


# 0 means malignant and 1 means benign
sns.pairplot(c_df, hue="target", vars = ["mean radius", "mean concavity", "mean smoothness"])
plt.show()


# ### Separating and splitting dataset

# In[152]:


# Separating features and labels
X = c_df.loc[:,c_df.columns!="target"]
y = c_df["target"]


# In[153]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


# ### Building the Classifier

# In[149]:


gnb = GaussianNB()


# In[154]:


gnb.fit(X_train, y_train)


# In[156]:


y_pred = gnb.predict(X_train)


# In[158]:


accuracy_score(y_pred, y_train)


# ### Hyperparameter Tuning
# 
# First doing manual hyperparameter tuning by visualizing the results, and later trying out automatic options to compare the results
# 
# - The hyperparameter `var_smoothing` of GuassianNB will be optimized.
# - The range was chosen by studying work of [EloquentML](https://eloquentarduino.github.io/2020/08/eloquentml-grows-its-family-of-classifiers-gaussian-naive-bayes-on-arduino/)

# In[164]:


predicion_array = {'train_acc':[], 'test_acc':[], 'train_error':[], 'test_error':[]}
for i in range(-7,1):
    gnb = GaussianNB(var_smoothing=pow(10, i))
    gnb.fit(X_train, y_train)
    train_pred = gnb.predict(X_train)
    test_pred = gnb.predict(X_test)
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    predicion_array['train_acc'].append(train_accuracy)
    predicion_array['test_acc'].append(test_accuracy)
    predicion_array['train_error'].append(1 - train_accuracy)
    predicion_array['test_error'].append(1 - test_accuracy)


# In[165]:


predicion_array


# In[166]:


x_axis = [x for x in range(-1,7)]
plt.figure(figsize=(16,10), dpi= 80)
no_of_features = list(range(1,len(X.columns)+1))
plt.plot(x_axis, predicion_array['train_acc'], color='tab:red', label='Train Accuracy')
plt.plot(x_axis, predicion_array['test_acc'], color="steelblue", label='Test Accuracy')
plt.ylabel("Accuracy", fontsize=12)
plt.xlabel("Var_smoothing", fontsize=12)
plt.yticks(fontsize=12, alpha=.9)
plt.title("Train and Test Accuracy for varying Var_smoothing (Gaussian NB)", fontsize=16)
plt.legend()
plt.show()


# In[168]:


x_axis = [x for x in range(-1,7)]
plt.figure(figsize=(16,10), dpi= 80)
no_of_features = list(range(1,len(X.columns)+1))
plt.plot(x_axis, predicion_array['train_error'], color='tab:green', label='Train Error')
plt.plot(x_axis, predicion_array['test_error'], color="tab:red", label='Test Error')
plt.ylabel("Error", fontsize=12)
plt.xlabel("Var_smoothing", fontsize=12)
plt.yticks(fontsize=12, alpha=.9)
plt.title("Train and Test Error for varying Var_smoothing (Gaussian NB)", fontsize=16)
plt.legend()
plt.show()

