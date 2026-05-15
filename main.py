import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load dataset
data = pd.read_csv("student_data.csv")

print("Dataset:")
print(data.head())

# Features (inputs)
X = data[['study_hours', 'attendance', 'sleep_hours', 'assignments']]

# Target (output)
y = data['marks']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict test data
predictions = model.predict(X_test)

# Error calculation
error = mean_absolute_error(y_test, predictions)

print("\nMean Absolute Error:", error)

# User prediction
study_hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance percentage: "))
sleep_hours = float(input("Enter sleep hours: "))
assignments = float(input("Enter assignments completed: "))

result = model.predict([[study_hours, attendance, sleep_hours, assignments]])

print("\nPredicted Marks:", round(result[0], 2))

# Graph
plt.scatter(data['study_hours'], data['marks'])
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")

plt.show()