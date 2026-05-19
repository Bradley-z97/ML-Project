import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
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
    # 'Cabin' field is irrelevant to our model
    df = df.drop(['Cabin'], axis=1)
    # Only 2 NA values, these will be dropped
    df = df.drop(['Embarked'], axis=1)
    df = df.drop(['Name'], axis=1)
    df = df.drop(['Ticket'], axis =1)
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


if __name__ == '__main__':
    clean_df = data_cleaning(df)
    correlation_matrix(clean_df)
    