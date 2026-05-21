import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import root_mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

df = pd.read_csv('Titanic-Dataset.csv')

def interpret(data):
    print(data.head())
    print(data.describe())
    print(data.info)
    print(data.isnull().sum())

def data_cleaning(df):
    df = df.copy()
    # Missing age will be replaced by the average
    avg = df['Age'].mean().round()
    df['Age'] = df['Age'].fillna(avg)
    # Drop irrelevant columns
    df = df.drop(['PassengerId',
                  'Name',
                  'Ticket',
                  'Embarked',
                  'Cabin'], axis=1)
    # Male = 1, Female = 0
    df['Sex'] = (df['Sex'] == 'male').astype(int)
    return df

def correlation_matrix(df):
    matrix = df.corr()

    plt.figure(figsize=(10,8))
    sns.heatmap(matrix,
                annot=True,
                cmap='magma')
    plt.savefig('plot.png')

def log_reg(df):
    model = LogisticRegression() 
    
    X = df[['Sex', 'Pclass', 'Fare', 'Age']]
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    return y_test ,y_pred

def neural_network(df):
    model = MLPClassifier((64, 32), random_state=42)
    
    X = df[['Sex', 'Pclass', 'Fare', 'Age']]
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    return y_test ,y_pred

def knn(df):
    model = KNeighborsClassifier(n_neighbors=5)

    X = df[['Sex', 'Pclass', 'Fare', 'Age']]
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return y_test, y_pred

def check_metrics(actual, pred):
    acc = accuracy_score(actual, pred)
    rmse = root_mean_squared_error(actual, pred)

    string = f'Accuracy Score: {acc}\nRMSE: {rmse}'
    return string


if __name__ == '__main__':
    clean_df = data_cleaning(df) 
    correlation_matrix(clean_df)
    # Logistic regression
    actual, prediction = log_reg(clean_df)
    print('---Logistic Regression---')
    print(check_metrics(actual, prediction))
    print('\n')
    # Neural network
    actual, prediction = neural_network(clean_df)
    print('---Neural Network---')
    print(check_metrics(actual, prediction))
    print('\n')
    # K Nearest Neighbours
    actual, prediction = knn(clean_df)
    print('---K Nearest Neighbours---')
    print(check_metrics(actual, prediction))
    
