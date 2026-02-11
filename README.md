# BLENDED_LERNING
# Implementation-of-Multiple-Linear-Regression-Model-with-Cross-Validation-for-Predicting-Car-Prices

## AIM:
To write a program to predict the price of cars using a multiple linear regression model and evaluate the model performance using cross-validation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and preprocess the car dataset (clean data, encode categorical features, normalize if needed) and split features (X) and target price (y).
2. Initialize a Multiple Linear Regression model and set up (k)-fold cross-validation.
3. For each fold, train the model on (k-1) folds and predict prices on the remaining fold.
4. Compute and average performance metrics (e.g., RMSE, (R^2)) across folds to evaluate the model.


## Program:

Developed by: DHARANI B
RegisterNumber:  212225230053
```
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

#1. Load and prepare data
data = pd.read_csv('CarPrice_Assignment (1).csv')

#Simple preprocessing
data = data.drop(['car_ID', 'CarName'], axis=1)  # Remove unnecessary columns
data = pd.get_dummies(data, drop_first=True)     # Handle categorical variables

#2. Split data
X = data.drop('price', axis=1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Evaluate with cross-validation (simple version)
print("Name: dharani b ")
print("Reg. No: 212225230053")
print("\n=== Cross-Validation ===")
cv_scores = cross_val_score(model, X, y, cv=5)
print("Fold R² scores:", [f"{score:.4f}" for score in cv_scores])
print(f"Average R²: {cv_scores.mean():.4f}")

# 5. Test set evaluation
y_pred = model.predict(X_test)
print("\nTest Set Performance")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")

print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"R2: {r2_score(y_test, y_pred):.4f}")
# 6. Visualization
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.grid(True)
plt.show()
```
## Output:
<img width="833" height="277" alt="Screenshot 2026-02-11 111323" src="https://github.com/user-attachments/assets/f9492ec8-7d7f-4282-aa00-bceafd4d70b4" />
<img width="965" height="670" alt="Screenshot 2026-02-11 111346" src="https://github.com/user-attachments/assets/9597e99c-5a52-4bcc-aa35-7b59014b9b70" />


## Result:
Thus, the program to implement the multiple linear regression model with cross-validation for predicting car prices is written and verified using Python programming.
