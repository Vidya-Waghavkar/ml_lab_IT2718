import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

     

data = {
    'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Pass':       [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)
print("=== Original Dataset ===")
print(df)

     

X = df[['StudyHours']]  # Independent variable
y = df['Pass']          # Dependent variable

     

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
     

model = LogisticRegression()
model.fit(X_train, y_train)

     

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

     

print("\n=== Model Evaluation ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

     

X_values = np.linspace(0, 10, 100).reshape(-1, 1)
Y_prob = model.predict_proba(X_values)[:, 1]

plt.figure(figsize=(7,5))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X_values, Y_prob, color='red', linewidth=2, label='Logistic Curve')
plt.title('Logistic Regression (Study Hours vs Pass Probability)')
plt.xlabel('Study Hours')
plt.ylabel('Probability of Passing')
plt.legend()
plt.grid(True)
plt.show()