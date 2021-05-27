# -*- coding: utf-8 -*-
"""
Created on Thu May 27 19:19:03 2021

@author: Pierre
"""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from joblib import dump, load

import streamlit as st


@st.cache
def get_model(choix, X_train, y_train, X_test, y_test):

    options = ['Regression Logistique', 'KNN', 'Decision Tree']
    if choix == options[0]:
        model = load("logreg.joblib")
        
        score = model.score(X_test, y_test)
        
    if choix == options[1]:
        model = load("knn.joblib")
        
        score = model.score(X_test, y_test)
        
    if choix == options[2]:
        model = load("tree.joblib")
        
        score = model.score(X_test, y_test)
        
    return model, score