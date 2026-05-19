import pandas as pd
import numpy as np
import matplotlib as plt

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
    df = df.dropna(subset=['Embarked'])
    return df


if __name__ == '__main__':
    clean_df = data_cleaning(df)
    print(clean_df.head())
