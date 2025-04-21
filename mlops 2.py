import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import pickle

# Step 1: Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Step 2: Split into training and testing sets
X = df[iris.feature_names]
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train initial model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Step 4: Evaluate the initial model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Initial Model Accuracy: {accuracy:.2f}")

# Step 5: Save initial model (Version 1)
with open('model_v1.pkl', 'wb') as f:
    pickle.dump(model, f)

print("model_v1.pkl saved successfully!")

# Step 6: Hyperparameter tuning (optional)
param_grid = {'C': [0.1, 1, 10]}
grid_search = GridSearchCV(LogisticRegression(max_iter=200), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best tuned model
best_model = grid_search.best_estimator_

# Step 7: Evaluate the best model
y_pred_best = best_model.predict(X_test)
best_accuracy = accuracy_score(y_test, y_pred_best)
print(f"Tuned Model Accuracy: {best_accuracy:.2f}")

# Step 8: Save best tuned model (Version 2)
with open('model_v2.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("model_v2.pkl saved successfully!")