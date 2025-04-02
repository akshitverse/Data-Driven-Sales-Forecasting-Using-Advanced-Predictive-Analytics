from flask import Flask, jsonify, render_template, request
import pickle
import pandas as pd
import numpy as np
import mysql.connector

# Load the trained model with feature names
model_path = 'sales_random_forest.pkl'
with open(model_path, 'rb') as file:
    data = pickle.load(file)

model = data["model"]  # Load the trained model
feature_names = data["features"]  # Load the saved feature names

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
        # Extracting input values
        quantity = int(request.form["Quantity"])
        discount = float(request.form["Discount"])
        profit = float(request.form["Profit"])
        state = request.form["State"]

        # Create DataFrame for input
        user_df = pd.DataFrame([{
            "Quantity": quantity,
            "Discount": discount,
            "Profit": profit,
            "State": state
        }])

        # One-hot encode state
        user_df = pd.get_dummies(user_df, columns=["State"])

        # Ensure all required columns exist
        for col in feature_names:
            if col not in user_df.columns:
                user_df[col] = 0  # Add missing columns with 0

        # Reorder columns to match training data
        user_df = user_df[feature_names]

        # Convert to numpy array for prediction
        user_input_array = user_df.values

        # Make prediction
        prediction = model.predict(user_input_array)[0]
        prediction = round(prediction, 2)

        # Store data in MySQL
        sql = "INSERT INTO Sales_Data (quantity, discount, profit, state, predicted_sales) VALUES (%s, %s, %s, %s, %s)"
        values = (quantity, discount, profit, state, prediction)
        cursor.execute(sql, values)
        db.commit()

        return render_template('index.html', prediction_value=f'Prediction: {prediction}')

    except Exception as e:
        return render_template('index.html', prediction_value=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True, port=4400)
