import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Generate dummy dataset
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Feature
y = 2.5 * X + np.random.randn(100, 1) * 2  # Target with some noise

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "regression_model.pkl")
