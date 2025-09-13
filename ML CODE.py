import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load dataset
dataset = pd.read_csv('crop_yield.csv')

# Separate features and target
X = dataset.drop('Yield', axis=1)
y = dataset['Yield']

# One-hot encode categorical features
categorical_cols = X.select_dtypes(include=['object']).columns
X = pd.get_dummies(X, columns=categorical_cols)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")

# Feature importance
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
print("\nTop 10 Important Features:")
print(feature_importances.sort_values(ascending=False).head(10))

# Save model
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model exported successfully as random_forest_model.pkl")
