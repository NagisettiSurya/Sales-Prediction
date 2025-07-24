import pandas as pd
from sklearn.linear_model import LinearRegression

# 1. Prepare Data
tv_ads = [5, 15, 25, 35, 45, 55]     # ad spend in $1000s
sales = [7, 14, 20, 27, 34, 40]      # sales in $1000s

data = pd.DataFrame({'TV_Ads': tv_ads, 'Sales': sales})

# 2. Train Model
X = data[['TV_Ads']]
y = data['Sales']

regressor = LinearRegression()
regressor.fit(X, y)

# 3. Make Prediction
test_ad_spend = 60  # $60,000 in ads
predicted_sales = regressor.predict([[test_ad_spend]])

print(f"Predicted Sales for ${test_ad_spend}k in TV ads: ${predicted_sales[0]:.2f}k")

# 4. Visualize
import matplotlib.pyplot as plt

plt.scatter(tv_ads, sales, color='blue', label='Actual Data')
plt.plot(tv_ads + [test_ad_spend],
         list(regressor.predict(X)) + [predicted_sales[0]],
         color='red', label='Prediction')
plt.xlabel('TV Advertising Spend ($1000s)')
plt.ylabel('Sales ($1000s)')
plt.title('Sales Prediction Based on TV Advertising')
plt.legend()
plt.show()
