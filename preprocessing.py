# -*- coding: utf-8 -*-
"""
Created on Thu May 27 19:18:54 2021

@author: Pierre
"""
import pandas as pd

def generate_train_data():

    df = pd.read_csv("train.csv")
    
    
    X = df.drop('Survived', axis = 1)
    y = df['Survived']
    
    
    # Nettoyage
    
    X = X.drop(['Name', 'Ticket', 'Cabin'], axis = 1)
    X['Sex'] = X['Sex'].apply(lambda x : x == 'female')
    X['Age'] = X['Age'].fillna(X['Age'].mode()[0])
    X['Embarked'] = X['Embarked'].fillna(X['Embarked'].mode()[0])
    
    X = pd.get_dummies(X, columns = ['Embarked'])
    
    return X, y