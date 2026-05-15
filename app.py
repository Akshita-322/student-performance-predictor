import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("student_data.csv")

# Features and target
X = data[['study_hours', 'attendance', 'sleep_hours', 'assignments']]
y = data['marks']

# Train model
model = LinearRegression()
model.fit(X, y)

# Streamlit UI
st.title("🎓 Student Performance Prediction System")

st.write("Enter student details to predict marks.")

study_hours = st.number_input("Study Hours", min_value=0.0, max_value=24.0)
attendance = st.number_input("Attendance Percentage", min_value=0.0, max_value=100.0)
sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0)
assignments = st.number_input("Assignments Completed", min_value=0.0)

# Predict button
if st.button("Predict Marks"):

    prediction = model.predict([[study_hours, attendance, sleep_hours, assignments]])

    st.success(f"Predicted Marks: {round(prediction[0], 2)}")