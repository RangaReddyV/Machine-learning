# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 14:05:40 2018

@author: rreddyv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder

## to import the data
train_df = pd.read_csv('train_data.csv',index_col=0)
test_df = pd.read_csv('test_data.csv',index_col=0)
col_names = train_df.columns.tolist()
# Encoding catogiracl data and missing data
         encode_features = ['Gender','Married','Education','Self_Employed','Dependents','Loan_Status']

fillna_withmean = ['LoanAmount','Loan_Amount_Term']
fillna_withmostcommon = ['Dependents','Gender','Credit_History','Married','Self_Employed']


def transform_df(data):
    
    #Removing Loans_ID
    df = data #.drop('Loan_ID',axis=1) 
    
    # Filling NaN values 
    for feature in fillna_withmean:
        if feature in data.columns.values:
            df[feature] = df[feature].fillna(df[feature].mean()) 
        
    for feature in fillna_withmostcommon:
        if feature in data.columns.values:
            df[feature] = df[feature].fillna(df[feature].value_counts().index[0])
    
    # Encoding Features
    for feature in encode_features:
        if feature in data.columns.values:
            le = LabelEncoder()
            df[feature] = le.fit_transform(df[feature])
    
    # Adding Applicant and Coapplicant Incomes as Household
    df['Household_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df = df.drop(['ApplicantIncome','CoapplicantIncome'],axis=1)
     
    # Transforming some other values   
    dummies = pd.get_dummies(df.Property_Area)
    df = pd.concat([df,dummies],axis=1)
    df = df.drop('Property_Area',axis=1)

    
    return df

train_df = transform_df(train_df)
test_df = transform_df(test_df)
                   
            
X_train = train_df.iloc[:, :-1]
Y_train = train_df.iloc[:,-1]
X_test = test_df 


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf = clf.fit(X_train, Y_train)

Y_test = clf.predict(X_test)           
Y_test = ['Y' if x==1 else 'N' for x in Y_test]
X_test['Loan_Status']=Y_test
X_test = X_test['Loan_Status']
X_test.to_csv('loans_submission.csv',sep=',',header=True)