import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

     

data = {
    'YearsExperience': [1.1, 1.3, 1.5, 2.0, 2.2, 2.9, 3.0, 3.2, 3.2, 3.7, 3.9, 4.0,
                        4.1, 4.5, 4.9, 5.1, 5.3, 5.9, 6.0, 6.8, 7.1, 7.9, 8.2, 8.7,
                        9.0, 9.5, 9.6, 10.3, 10.5],
    'Salary': [39343, 46205, 37731, 43525, 39891, 56642, 60150, 54445, 64445, 57189,
               63218, 55794, 56957, 61111, 67938, 66029, 83088, 81363, 93940, 91738,
               98273, 101302, 113812, 109431, 105582, 116969, 112635, 122391, 121872]
}
df = pd.DataFrame(data)
print("Dataset Loaded Successfully\n")
print(df.head())

     

plt.figure(figsize=(8,5))
sns.scatterplot(x='YearsExperience', y='Salary', data=df, color='blue')
plt.title("Scatter Plot – Salary vs Years of Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

     

X = df[['YearsExperience']]
y = df['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
     

print(f" Intercept (a): {model.intercept_:.2f}")
print(f" Coefficient (b): {model.coef_[0]:.2f}")

     

print(f"\nEquation of regression line: Salary = {model.intercept_:.2f} + ({model.coef_[0]:.2f} × YearsExperience)")
     

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.3f}")

     

plt.figure(figsize=(8,5))
sns.scatterplot(x='YearsExperience', y='Salary', data=df, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.title("Linear Regression – Salary vs Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.show()
 

new_exp = pd.DataFrame({'YearsExperience': [6]})
predicted_salary = model.predict(new_exp)[0]
print(f"\n Predicted Salary for 6 years of experience = ₹{predicted_salary:.2f}")