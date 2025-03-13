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

- **Mean Squared Error (MSE)**: 97569.72294013854
- **Root Mean Squared Error (RMSE)**:312.3615260241545
- **R-squared (R²)**: 0.895576840810998

### Feature Importance

The following features were found to be most important in predicting daily revenue:

1. Average order value
2. Number of order per day


## Potential Improvements

- **Feature Engineering**: Create new features or transform existing ones to capture more complex relationships.
- **Model Selection**: Try other regression models like polynomial regression, ridge regression, or lasso regression to see if they perform better.
- **Hyperparameter Tuning**: Use techniques like cross-validation and grid search to fine-tune the model's hyperparameters.
- **Handling Outliers**: Identify and handle outliers that may be affecting the model's performance.

