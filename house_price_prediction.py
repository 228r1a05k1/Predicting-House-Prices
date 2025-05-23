import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_csv("house_data.csv")

# Visualize the data
plt.scatter(df['sqft'], df['price'], color='blue', label='Square Footage')
plt.scatter(df['bedrooms'], df['price'], color='red', label='Bedrooms')
plt.title("House Price vs Features")
plt.xlabel("Feature Value")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

# Prepare features and labels
X = df[['bedrooms', 'sqft']]  # Features
y = df['price']               # Target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

print("=== Model Evaluation ===")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R-squared (R²):", r2_score(y_test, y_pred))
print("Coefficients (bedrooms, sqft):", model.coef_)
print("Intercept:", model.intercept_)

# Predict a new house price
new_data = pd.DataFrame({
    'bedrooms': [3],
    'sqft': [1400]
})
predicted_price = model.predict(new_data)
print(f"\nPredicted price for 3 bedrooms, 1400 sqft: ${predicted_price[0]:,.2f}")
