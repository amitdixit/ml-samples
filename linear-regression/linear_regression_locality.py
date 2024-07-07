import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('mumbai_house_prices.csv')



# Define the column transformer with one-hot encoding for 'region'
column_transformer = ColumnTransformer(
    transformers=[
        ('region', OneHotEncoder(), ['region'])
    ], remainder='passthrough'  # Keep the 'area' column as is
)


# Create a pipeline with the column transformer and the linear regression model
pipeline = make_pipeline(column_transformer, LinearRegression())

# Fit the model using 'region' and 'area' as features
pipeline.fit(df[['region', 'area']], df['price_in_cr'])

# Example region and area for prediction
example_region = 'Andheri West'
example_area = 500

# Transform the example input to the appropriate format
example_input = pd.DataFrame({'region': [example_region], 'area': [example_area]})

# Predict the price
predicted_price = pipeline.predict(example_input)
print(f"Predicted price for region {example_region} and area {example_area} sq ft: {predicted_price[0]:.2f} Cr")

# Print the model coefficients and intercept
model = pipeline.named_steps['linearregression']
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")

# Plot the data for each region
localities = df['region'].unique()

# Set up the plot
plt.figure(figsize=(14, 10))

# Loop through each region to create separate scatter plots and regression lines
# for region in localities:
#     subset = df[df['region'] == region]
#     plt.scatter(subset['area'], subset['price_in_cr'], label=region)
    
#     # Fit a linear regression model for each region
#     reg_region = LinearRegression()
#     reg_region.fit(subset[['area']], subset['price_in_cr'])
    
#     # Predict prices based on area for plotting the regression line
#     area_range = np.linspace(subset['area'].min(), subset['area'].max(), 100).reshape(-1, 1)
#     price_pred = reg_region.predict(area_range)
    
#     plt.plot(area_range, price_pred, label=f'{region} regression line')

# plt.xlabel('Area (sq ft)')
# plt.ylabel('Price (Cr)')
# plt.title('Area vs. Price for Different Localities')
# plt.legend()
# plt.show()

