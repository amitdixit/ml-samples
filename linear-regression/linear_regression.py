import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('mumbai_house_prices.csv')
# print(df)

# print(df['price_in_cr'].max())

max_price_row = df.loc[df['price_in_cr'].idxmax()]

print("Row with maximum price:\n", max_price_row)
# print(max_price_row)

print(df[['area']])


# plt.show(block = False)


reg = LinearRegression()
reg.fit(df[['area']],df.price_in_cr)

predicted_price = reg.predict([[500]])
print(f"Predicted price for area 500 sq ft: {predicted_price[0]:.2f} Cr")

# Print the model parameters
print(f"Coefficient: {reg.coef_[0]:.2f}")
print(f"Intercept: {reg.intercept_:.2f}")

plt.xlabel('area')
plt.ylabel('price (INR)')
plt.scatter(df['area'], df['price_in_cr'],color='blue',marker="+")
# Plot the regression line
plt.plot(df['area'], reg.predict(df[['area']]), color='red')
plt.title('Area vs. Price')
plt.show()
print(reg)




new_df = pd.read_csv('areas_to_predict.csv')
print(new_df)
new_p = reg.predict(new_df)

print(new_p)
new_df['price'] = new_p
new_df.to_csv('predicted_prices.csv', index=False)
# plt.plot(df.area,reg.predict(df[['area']]),color='blue')


# plt.show()