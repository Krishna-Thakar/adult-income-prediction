# Machine Learning Project: Adult Income Prediction

This project aims to predict whether an individual earns more than $50K per year based on various attributes.
The dataset used is the Adult Income dataset from the UCI Machine Learning Repository.

## Dataset

The dataset consists of demographic information about individuals and includes the following columns:
- Age
- Workclass
- Someid
- Education
- Education_num
- Marital_status
- Occupation
- Relationship
- Race
- Sex
- Capital-gain
- Capital-loss
- Hours_per_week
- Native_country
- Class (Target variable: >50K or <=50K)

## Libraries Used

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost

## Data Preprocessing

- Removed duplicates
- Filled missing values with the mode
- Stripped leading and trailing whitespaces
- Label encoding for categorical variables

## Model Training and Evaluation

The following models were trained and evaluated:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Naive Bayes
- K-Nearest Neighbors
- XGBoost
- AdaBoost
- Gradient Boosting

## Results

Each model's accuracy on the test set is printed to the console.

## How to Run

1. Install the necessary libraries:
    ```shell
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost
    ```
2. Run the script:
    ```shell
    adult_income_prediction.py
    ```


