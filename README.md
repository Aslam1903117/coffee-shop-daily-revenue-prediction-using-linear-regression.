# Coffee Shop Daily Revenue Prediction

This project aims to predict the daily revenue of a coffee shop using a linear regression model. The dataset used is the "Coffee Shop Daily Revenue Prediction" dataset from Kaggle.

## Dataset Overview

The dataset contains the following columns:

- **Number_of_Customers_Per_Day**: The number of customers visiting the coffee shop each day.
- **Average_Order_Value**: The average value of orders placed by customers.
- **Operating_Hours_Per_Day**: The number of hours the coffee shop operates each day.
- **Number_of_Employees**: The number of employees working at the coffee shop.
- **Marketing_Spend_Per_Day**: The amount spent on marketing each day.
- **Location_Foot_Traffic**: The estimated foot traffic around the coffee shop's location.
- **Daily_Revenue**: The daily revenue of the coffee shop (target variable).


## Approach

### Step 1: Data Preprocessing

- Checked for missing values (none found).
- Ensured all data types were appropriate for analysis.

### Step 2: Exploratory Data Analysis (EDA)

- Created a correlation matrix to understand the relationships between variables.
- Used a pairplot to visualize the relationships between all pairs of variables.

### Step 3: Model Preparation

- Separated the features (X) from the target variable (y).
- Split the data into training and testing sets using `train_test_split` with a test size of 20% and `random_state=42` for reproducibility.

### Step 4: Model Training and Evaluation

- Trained a linear regression model on the training data.
- Made predictions on the test set and calculated performance metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R²).

### Step 5: Residual Analysis

- Calculated residuals and plotted them against predicted values to check for patterns and ensure the model's assumptions were met.
- Created a histogram of residuals to check for normality.

### Step 6: Feature Importance

- Analyzed the importance of each feature based on the model's coefficients.

## Results

### Model Performance Metrics

- **Mean Squared Error (MSE)**: [Insert MSE value here]
- **Root Mean Squared Error (RMSE)**: [Insert RMSE value here]
- **R-squared (R²)**: [Insert R² value here]

### Feature Importance

The following features were found to be most important in predicting daily revenue:

1. [Insert most important feature here]
2. [Insert second most important feature here]
3. [Insert third most important feature here]

![Feature Importance](feature_importance.png)

### Residual Analysis

![Residual Plot](residual_plot.png)

![Histogram of Residuals](histogram_residuals.png)

## Potential Improvements

- **Feature Engineering**: Create new features or transform existing ones to capture more complex relationships.
- **Model Selection**: Try other regression models like polynomial regression, ridge regression, or lasso regression to see if they perform better.
- **Hyperparameter Tuning**: Use techniques like cross-validation and grid search to fine-tune the model's hyperparameters.
- **Handling Outliers**: Identify and handle outliers that may be affecting the model's performance.

## Code

```python
# coffee_shop_revenue_prediction.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the coffee shop daily revenue dataset.

    Args:
    file_path (str): Path to the CSV file containing the dataset.

    Returns:
    pd.DataFrame: Preprocessed dataset.
    """
    # Load the dataset
    data = pd.read_csv(file_path)

    # Check for missing values
    print("Missing values:")
    print(data.isnull().sum())

    return data

def perform_eda(data):
    """
    Perform Exploratory Data Analysis (EDA) on the dataset.

    Args:
    data (pd.DataFrame): The dataset to analyze.

    Returns:
    None
    """
    # Correlation matrix
    correlation_matrix = data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig('correlation_matrix.png')
    plt.close()

    # Pairplot to visualize relationships between variables
    sns.pairplot(data)
    plt.savefig('pairplot.png')
    plt.close()

def prepare_features_and_target(data):
    """
    Prepare features and target variable for modeling.

    Args:
    data (pd.DataFrame): The dataset to prepare.

    Returns:
    tuple: X (features), y (target), X_train, X_test, y_train, y_test
    """
    # Define features and target
    X = data.drop('Daily_Revenue', axis=1)
    y = data['Daily_Revenue']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Display the shapes of the training and testing sets
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    return X, y, X_train, X_test, y_train, y_test

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    """
    Train a linear regression model and evaluate its performance.

    Args:
    X_train (pd.DataFrame): Training features.
    X_test (pd.DataFrame): Testing features.
    y_train (pd.Series): Training target.
    y_test (pd.Series): Testing target.

    Returns:
    tuple: Trained model, predictions, MSE, RMSE, R²
    """
    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate the model's performance metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"R-squared Score: {r2}")

    # Display the model's coefficients and intercept
    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)

    return model, y_pred, mse, rmse, r2

def analyze_residuals(y_test, y_pred):
    """
    Analyze the residuals of the model's predictions.

    Args:
    y_test (pd.Series): Actual target values.
    y_pred (np.array): Predicted target values.

    Returns:
    None
    """
    # Calculate residuals
    residuals = y_test - y_pred

    # Plot residuals
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.savefig('residual_plot.png')
    plt.close()

    # Histogram of residuals
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.title('Histogram of Residuals')
    plt.savefig('histogram_residuals.png')
    plt.close()

def plot_feature_importance(model, X):
    """
    Plot the feature importance based on the model's coefficients.

    Args:
    model (LinearRegression): Trained linear regression model.
    X (pd.DataFrame): Features used in the model.

    Returns:
    None
    """
    # Feature importance
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': abs(model.coef_)})
    feature_importance = feature_importance.sort_values('Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.savefig('feature_importance.png')
    plt.close()

def main():
    """
    Main function to run the entire analysis pipeline.
    """
    # Load and preprocess data
    data = load_and_preprocess_data('coffee-shop-daily-revenue-prediction-dataset.csv')

    # Perform EDA
    perform_eda(data)

    # Prepare features and target
    X, y, X_train, X_test, y_train, y_test = prepare_features_and_target(data)

    # Train and evaluate model
    model, y_pred, mse, rmse, r2 = train_and_evaluate_model(X_train, X_test, y_train, y_test)

    # Analyze residuals
    analyze_residuals(y_test, y_pred)

    # Plot feature importance
    plot_feature_importance(model, X)

if __name__ == "__main__":
    main()
