import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load your dataset
df = pd.read_csv('mumbai_house_prices.csv')

# Drop rows with missing values (assuming all columns used have been cleaned)
df.dropna(inplace=True)

# Define features and target
features = ['bhk', 'type', 'locality', 'area', 'region', 'status', 'age_new']
target = 'price_in_cr'

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Preprocessing pipeline
# One-hot encode categorical features and scale numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['type', 'locality', 'region', 'status']),
        ('num', StandardScaler(), ['bhk', 'area', 'age_new' ])
    ])

# Define the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))  # Example of using Random Forest Regressor
])

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Additional steps:
# - Hyperparameter tuning (GridSearchCV or RandomizedSearchCV)
# - Cross-validation for better generalization
# - Feature selection to identify the most important features

# Example of predicting with new data
new_data = pd.DataFrame({
    'bhk': [3],
    'type': ['Apartment'],
    'locality': ['Andheri West'],
    'area': [1000],
    'region': ['Western Suburbs'],
    'status': ['Ready to move'],
    'age_new': [0]
})
predicted_price = model.predict(new_data)
print(f"Predicted Price: {predicted_price[0]:.2f} Cr")
