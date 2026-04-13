# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset (you can use any CSV dataset)
# Example: House Price dataset
df = pd.read_csv("housing.csv")

# Display first 5 rows
print(df.head())

# Preprocessing
df = df.dropna()  # remove missing values

# Features (X) and Target (y)
X = df[['area', 'bedrooms', 'bathrooms']]   # multiple regression
y = df['price']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Plot (for simple regression using area only)
plt.scatter(X_test['area'], y_test, color='blue')
plt.plot(X_test['area'], y_pred, color='red')
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Linear Regression")
plt.show()