
"""
Machine Learning Project: Adult Income Prediction

This project aims to predict whether an individual earns more than $50K per year based on various attributes.
The dataset used is the Adult Income dataset from the UCI Machine Learning Repository.

Author: [Your Name]
Date: [Current Date]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Load dataset
header = ["age", "workclass", "someid", "education", "education_num", "marital_status", "occupation", 
          "relationship", "race", "sex", "capital-gain", "capital_loss", "hours_per_week", 
          "native_country", "class"]

df = pd.read_csv("adult.csv", header=None, names=header, na_values=[' ?'])

# Data Preprocessing
df.drop_duplicates(inplace=True)
cols = ["workclass", "occupation", "native_country"]
for col in cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].str.strip()

# Label Encoding
label = LabelEncoder()
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = label.fit_transform(df[col])

# Splitting the dataset
y = df['class']
X = df.drop("class", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Model Training and Evaluation
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": GaussianNB(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

# Evaluate each model
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy * 100:.2f}%")