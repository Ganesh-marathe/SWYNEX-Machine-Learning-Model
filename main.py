import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("Loading data...")
data = load_wine()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1
m1 = LogisticRegression(max_iter=10000)
m1.fit(X_train, y_train)
acc1 = accuracy_score(y_test, m1.predict(X_test))

# Model 2
m2 = RandomForestClassifier(random_state=42)
m2.fit(X_train, y_train)
acc2 = accuracy_score(y_test, m2.predict(X_test))

print(f"Logistic Regression Accuracy: {acc1:.4f}")
print(f"Random Forest Accuracy: {acc2:.4f}")
