import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
data = pd.read_csv("student_data.csv")
# Show dataset
st.subheader("📊 Student Dataset")
st.write(data)

# Features and target
X = data[['study_hours', 'attendance', 'sleep_hours', 'assignments']]
y = data['marks']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = r2_score(y_test, y_pred)
# Streamlit UI
st.title("🎓 Student Performance Prediction System")

st.write("Enter student details to predict marks.")
st.subheader("📌 Model Accuracy")

st.success(f"Model Accuracy Score: {round(accuracy * 100, 2)}%")


study_hours = st.number_input("Study Hours", min_value=0.0, max_value=24.0)
attendance = st.number_input("Attendance Percentage", min_value=0.0, max_value=100.0)
sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0)
assignments = st.number_input("Assignments Completed", min_value=0.0)

# Predict button
if st.button("Predict Marks"):

    prediction = model.predict([[study_hours, attendance, sleep_hours, assignments]])

    st.success(f"Predicted Marks: {round(prediction[0], 2)}")
    # Charts Section
st.subheader("📈 Data Visualization")

# Study Hours vs Marks
st.write("### Study Hours vs Marks")
st.line_chart(data[['study_hours', 'marks']])

# Attendance vs Marks
st.write("### Attendance vs Marks")
st.bar_chart(data[['attendance', 'marks']])