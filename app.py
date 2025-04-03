from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import mysql.connector
from datetime import datetime

# Load the trained model with feature names
model_path = 'sales_linear_regression.pkl'
with open(model_path, 'rb') as file:
    data = pickle.load(file)

model = data["model"]
feature_names = data["features"]

# Initialize Flask app
app = Flask(__name__)

# Configure MySQL connection
db = mysql.connector.connect(
    host="localhost",
    user="root",  
    password="starplatinum",  
    database="Sales"
)
cursor = db.cursor()

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Extract form values safely
        order_date_str = request.form.get("OrderDate")  # Get date as string
        quantity = request.form.get("Quantity", type=int)
        discount = request.form.get("Discount", type=float)
        profit = request.form.get("Profit", type=float)
        state = request.form.get("State")

        if not all([order_date_str, quantity, discount, profit, state]):
            return render_template('index.html', prediction_value="Error: Missing Input Fields")

        # Convert order date to datetime object
        order_date = datetime.strptime(order_date_str, "%Y-%m-%d")

        # Extract time-related features
        order_week = order_date.isocalendar()[1]  # ISO week number
        order_month = order_date.month
        order_quarter = (order_date.month - 1) // 3 + 1
        order_year = order_date.year

        # Create DataFrame for input
        user_df = pd.DataFrame([{
            "Quantity": quantity,
            "Discount": discount,
            "Profit": profit,
            "State": state,
            "Order_Week": order_week,
            "Order_Month": order_month,
            "Order_Quarter": order_quarter,
            "Order_Year": order_year
        }])

        # One-hot encode 'State'
        user_df = pd.get_dummies(user_df, columns=["State"])

        # Ensure all required columns exist
        for col in feature_names:
            if col not in user_df.columns:
                user_df[col] = 0  # Add missing columns with 0

        # Reorder columns to match model training data
        user_df = user_df[feature_names]

        # Convert to numpy array for prediction
        user_input_array = user_df.values

        # Make prediction
        prediction = model.predict(user_input_array)[0]
        prediction = round(prediction, 2)

        # Store data in MySQL
        sql = """INSERT INTO Sales_Data 
                 (order_date, quantity, discount, profit, state, order_week, order_month, order_quarter, order_year, predicted_sales) 
                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
        values = (order_date_str, quantity, discount, profit, state, order_week, order_month, order_quarter, order_year, prediction)
        cursor.execute(sql, values)
        db.commit()

        return render_template('index.html', prediction_value=f'Prediction: {prediction}')

    except Exception as e:
        return render_template('index.html', prediction_value=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True, port=4400)
