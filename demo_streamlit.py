# -*- coding: utf-8 -*-
"""
Created on Thu May 27 18:42:23 2021

@author: Pierre
"""

import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

st.title("Démo Streamlit Mar21 DA DS")

df = pd.read_csv("train.csv", index_col = 'PassengerId')

st.write(df)

X = df.drop('Survived', axis = 1)
y = df['Survived']


# Nettoyage

X = X.drop(['Name', 'Ticket', 'Cabin'], axis = 1)
X['Sex'] = X['Sex'].apply(lambda x : x == 'female')
X['Age'] = X['Age'].fillna(X['Age'].mode()[0])
X['Embarked'] = X['Embarked'].fillna(X['Embarked'].mode()[0])

st.write(X.isna().sum())



