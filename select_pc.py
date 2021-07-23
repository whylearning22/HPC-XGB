#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 19:10:54 2021

@author: luca
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import seed
#import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import csv
import shap
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import pearsonr
from sklearn import preprocessing
from sklearn.model_selection import LeaveOneGroupOut




#tot = tot[tot['annonascita'].notna()]


X=pd.read_csv("X_0803_onlypat.csv")
y=pd.read_csv("Y_0803.csv")


ynew=y[y['priorita'].isin([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24,25,26])].copy()
Xnew=X[y['priorita'].isin([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24,25,26])].copy()

ynew[ynew['priorita'].isin([1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24,25,26])]=100
ynew[ynew['priorita'].isin([8])]=200

ynew=ynew/100




ynew.to_csv('A/Y_2403_a.csv',index=False)
Xnew.to_csv('A/X_2403_a.csv',index=False)

# #ages b1
# Xnew=X[y['priorita'].isin([26,20])].copy()
# ynew=y[y['priorita'].isin([26,20])].copy()


# # ynew[ynew['priorita'].isin([26])]=100
# # ynew[ynew['priorita'].isin([20])]=200

# # ynew=ynew/100


# ynew.to_csv('dati/Y_2403_b1.csv',index=False)
# Xnew.to_csv('dati/X_2403_b1.csv',index=False)


#ages b2

#Xnew=X[y['priorita'].isin([25,19])].copy()
#ynew=y[y['priorita'].isin([25,19])].copy()


# ynew[ynew['priorita'].isin([25])]=100
# ynew[ynew['priorita'].isin([19])]=200

# ynew=ynew/100


#ynew.to_csv('Y_2403_b2.csv',index=False)
#Xnew.to_csv('X_2403_b2.csv',index=False)

#age b3

#Xnew=X[y['priorita'].isin([24,22,18,16])].copy()
#ynew=y[y['priorita'].isin([24,22,18,16])].copy()


# ynew[ynew['priorita'].isin([22])]=100
# ynew[ynew['priorita'].isin([18])]=200
# ynew[ynew['priorita'].isin([16])]=300


# ynew=ynew/100


#ynew.to_csv('dati/Y_2403_b3.csv',index=False)
#Xnew.to_csv('dati/X_2403_b3.csv',index=False)

#age b4
#Xnew=X[y['priorita'].isin([21,17,15])].copy()
#ynew=y[y['priorita'].isin([21,17,15])].copy()


# ynew[ynew['priorita'].isin([21])]=100
# ynew[ynew['priorita'].isin([17])]=200
# ynew[ynew['priorita'].isin([15])]=300


# ynew=ynew/100


#ynew.to_csv('dati/Y_2403_b4.csv',index=False)
#Xnew.to_csv('dati/X_2403_b4.csv',index=False)

#age b5

# Xnew=X[y['priorita'].isin([14,13,12,11,10,9])].copy()
# ynew=y[y['priorita'].isin([14,13,12,11,10,9])].copy()


# # ynew[ynew['priorita'].isin([14])]=100
# # ynew[ynew['priorita'].isin([13])]=200
# # ynew[ynew['priorita'].isin([12])]=300
# # ynew[ynew['priorita'].isin([11])]=400
# # ynew[ynew['priorita'].isin([10])]=500
# # ynew[ynew['priorita'].isin([9])]=600


# # ynew=ynew/100


# ynew.to_csv('dati/Y_2403_b5.csv',index=False)
# Xnew.to_csv('dati/X_2403_b5.csv',index=False)

#age b6

#Xnew=X[y['priorita'].isin([7,6,5,4,2])].copy()
#ynew=y[y['priorita'].isin([7,6,5,4,2])].copy()


# ynew[ynew['priorita'].isin([7])]=100
# ynew[ynew['priorita'].isin([6])]=200
# ynew[ynew['priorita'].isin([5])]=300
# ynew[ynew['priorita'].isin([4])]=400
# ynew[ynew['priorita'].isin([2])]=500


# ynew=ynew/100


#ynew.to_csv('Y_2403_b6.csv',index=False)
#Xnew.to_csv('X_2403_b6.csv',index=False)

#age b7

# Xnew=X[y['priorita'].isin([3,1])].copy()
# ynew=y[y['priorita'].isin([3,1])].copy()


# # ynew[ynew['priorita'].isin([3])]=100
# # ynew[ynew['priorita'].isin([1])]=200

# ynew.to_csv('Y_2403_b7.csv',index=False)
# Xnew.to_csv('X_2403_b7.csv',index=False)



#age b7

# Xnew=X[y['priorita'].isin([3,1])].copy()
# ynew=y[y['priorita'].isin([3,1])].copy()


# # ynew[ynew['priorita'].isin([3])]=100
# # ynew[ynew['priorita'].isin([1])]=200

# ynew.to_csv('Y_2403_b7.csv',index=False)
# Xnew.to_csv('X_2403_b7.csv',index=False)
