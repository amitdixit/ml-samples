import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import randint

# Sample Data
data = {
    'bhk': [3, 2, 3, 2, 4, 3, 4, 2, 3, 4],
    'region': ['Andheri West', 'Naigaon East', 'Andheri West', 'Naigaon East', 'Andheri West', 'Naigaon East', 'Andheri West', 'Naigaon East', 'Andheri West', 'Naigaon East'],
    'locality': ['Lak And Hanware The Residency Tower', 'Radheya Sai Enclave Building No 2', 'Lak And Hanware The Residency Tower', 'Radheya Sai Enclave Building No 2', 'Lak And Hanware The Residency Tower', 'Radheya Sai Enclave Building No 2', 'Lak And Hanware The Residency Tower', 'Radheya Sai Enclave Building No 2', 'Lak And Hanware The Residency Tower', 'Radheya Sai Enclave Building No 2'],
    'age_new': [0, 0, 1, 1, 0, 1, 1, 0, 0, 1],
    'area': [685, 640, 700, 600, 750, 670, 720, 610, 680, 730],
    'price_in_cr': [2.5, 0.5251, 2.6, 0.55, 3.0, 0.57, 2.8, 0.53, 2.4, 2.9]
}
df = pd.DataFrame(data)

# Prepare Data
X = df[['bhk', 'region', 'locality', 'age_new', 'area']]
y = df['price_in_cr']

# Encode categorical variables
encoder_region = OneHotEncoder()
encoder_locality = OneHotEncoder()

# Fit encoders separately
X_encoded_region = encoder_region.fit_transform(X[['region']]).toarray()
X_encoded_locality = encoder_locality.fit_transform(X[['locality']]).toarray()

X_encoded_region_df = pd.DataFrame(X_encoded_region, columns=encoder_region.get_feature_names_out(['region']))
X_encoded_locality_df = pd.DataFrame(X_encoded_locality, columns=encoder_locality.get_feature_names_out(['locality']))

# Combine encoded columns with the rest of the data
X = X.drop(columns=['region', 'locality']).reset_index(drop=True)
X = pd.concat([X, X_encoded_region_df, X_encoded_locality_df], axis=1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Randomized Search for Hyperparameter Tuning for RandomForestRegressor
param_dist = {
    'n_estimators': randint(10, 100),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['auto', 'sqrt', 'log2']
}

rf_model = RandomForestRegressor(random_state=42)
random_search = RandomizedSearchCV(rf_model, param_distributions=param_dist, n_iter=50, cv=5, verbose=1, n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

# Best RandomForestRegressor model
best_rf_model = random_search.best_estimator_

# Make predictions with RandomForestRegressor
rf_pred = best_rf_model.predict(X_test)

# Evaluate the RandomForestRegressor model
rf_mse = mean_squared_error(y_test, rf_pred)
print(f'Random Forest Regressor Mean Squared Error: {rf_mse}')

# Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions with Linear Regression
lr_pred = lr_model.predict(X_test)

# Evaluate the Linear Regression model
lr_mse = mean_squared_error(y_test, lr_pred)
print(f'Linear Regression Mean Squared Error: {lr_mse}')

# Example Prediction for both models
example = pd.DataFrame({'bhk': [2], 'age_new': [0], 'area': [650], 'region': ['Andheri West'], 'locality': ['Lak And Hanware The Residency Tower']})

# Encode the example with the same encoders used in training
example_encoded_region = encoder_region.transform(example[['region']]).toarray()
example_encoded_locality = encoder_locality.transform(example[['locality']]).toarray()

example_encoded_region_df = pd.DataFrame(example_encoded_region, columns=encoder_region.get_feature_names_out(['region']))
example_encoded_locality_df = pd.DataFrame(example_encoded_locality, columns=encoder_locality.get_feature_names_out(['locality']))

example = example.drop(columns=['region', 'locality']).reset_index(drop=True)
example = pd.concat([example, example_encoded_region_df, example_encoded_locality_df], axis=1)

rf_price_prediction = best_rf_model.predict(example)
print(f'Random Forest Regressor Predicted Price: {rf_price_prediction[0]} Cr')

lr_price_prediction = lr_model.predict(example)
print(f'Linear Regression Predicted Price: {lr_price_prediction[0]} Cr')
