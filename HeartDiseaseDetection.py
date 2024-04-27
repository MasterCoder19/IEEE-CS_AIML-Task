# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data
data = pd.read_csv("heart_disease_data.csv")

# Feature selection (choose relevant features for prediction)
features = ["age", "sex", "cp", "trestbps", "chol", "thalach", "fbs", "restecg", "exang", "oldpeak",	"slope",	"ca",	"thal"]
X = data[features]
y = data["target"]  # target variable (presence or absence of heart disease)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model (Random Forest is an example, choose the best fit for your data)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Make predictions on test set
y_pred = model.predict(X_test)

# Evaluate model performance (accuracy, precision, recall etc.)
from sklearn.metrics import accuracy_score, precision_score, recall_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# Use the trained model for new data points
new_data = pd.DataFrame({
    "age": [35],  # Replace with new data
    "sex": [0],  # Replace with new data
    "cp": [3], # Replace with new data
    "trestbps": [130],  # Replace with new data
    "chol": [200],  # Replace with new data
    "thalach": [140],  # Replace with new data
    "fbs": [1], # Replace with new data
    "restecg": [2], # Replace with new data
    "exang": [0], # Replace with new data
    "oldpeak": [1],	# Replace with new data
    "slope": [1], # Replace with new data
    "ca": [2], # Replace with new data
    "thal": [2] # Replace with new data
})

new_prediction = model.predict(new_data)[0]

if new_prediction == 1:
  print("This person is predicted to have high risk of heart disease.")
else:
  print("This person is predicted to have low risk of heart disease based on the model.")
