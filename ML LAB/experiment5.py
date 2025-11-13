import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # first two features
y = iris.target

# Use only two classes for 2D visualization
X = X[y != 2]
y = y[y != 2]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
model = SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("=== Model Evaluation ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot function
def plot_svm_decision_boundary(X, y, model):
    plt.figure(figsize=(7,5))
    plt.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm', s=30)
    
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    xx = np.linspace(xlim[0], xlim[1], 100)
    yy = np.linspace(ylim[0], ylim[1], 100)
    XX, YY = np.meshgrid(xx, yy)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)
    
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1],
               alpha=0.8, linestyles=['--', '-', '--'])
    
    ax.scatter(model.support_vectors_[:, 0],
               model.support_vectors_[:, 1],
               s=100, linewidth=1, facecolors='none', edgecolors='k')
    
    plt.title("SVM Decision Boundary (Setosa vs Versicolor)")
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    plt.show()

plot_svm_decision_boundary(X, y, model)
