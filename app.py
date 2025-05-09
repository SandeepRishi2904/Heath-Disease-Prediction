from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import pickle
from flask_cors import CORS
import warnings
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app) # Enable CORS for all domains

# Load all models
with open('D:\\Sandeep Rishi J B\\Hackathon\\Techathon\\Multiple Disease Prediction System\\saved models\\diabetes_model.sav', 'rb') as f:
    diabetes_model = pickle.load(f)

with open('D:\\Sandeep Rishi J B\\Hackathon\\Techathon\\Multiple Disease Prediction System\\saved models\\heart_disease_model.sav', 'rb') as f:
    heart_model = pickle.load(f)

with open('D:\\Sandeep Rishi J B\\Hackathon\\Techathon\\Multiple Disease Prediction System\\saved models\\parkinsons_model.sav', 'rb') as f:
    parkinsons_model = pickle.load(f)

@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    try:
        # Log the incoming request data
        data = request.get_json()
        logging.debug(f"Received data: {data}")

        input_data = data['input']
        logging.debug(f"Input data: {input_data}")

        # Define the feature names as used during model training
        feature_names = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]

        # Convert input data to a pandas DataFrame with feature names
        input_df = pd.DataFrame([input_data], columns=feature_names)
        logging.debug(f"DataFrame columns: {input_df.columns.tolist()}")
        logging.debug(f"DataFrame data: {input_df.to_dict()}")

        # Suppress the specific UserWarning about feature names
        warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")

        # Make prediction using the DataFrame directly
        prediction = diabetes_model.predict(input_df)[0]

        # Convert prediction to human-readable result
        is_diabetic = prediction == 1

        result = {
            'result': 'The model predicts you may be diabetic' if is_diabetic
                     else 'The model predicts you are not diabetic'
        }

        return jsonify(result)

    except Exception as e:
        logging.error(f"Error in predict_diabetes: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/predict/heart', methods=['POST'])
def predict_heart():
    try:
        # Get the input data from the POST request
        data = request.get_json()
        input_values = data['input']

        # Enforce sex encoding: male=1, female=0
        # Assuming sex is the second input (index 1) as per frontend form
        sex_value = input_values[1]
        if isinstance(sex_value, str):
            sex_value_lower = sex_value.lower()
            if sex_value_lower == 'male':
                input_values[1] = 1.0
            elif sex_value_lower == 'female':
                input_values[1] = 0.0
            else:
                # If unknown string, raise error
                return jsonify({"error": "Invalid sex value"}), 400
        else:
            # Convert to float and ensure 0 or 1
            sex_float = float(sex_value)
            if sex_float not in (0.0, 1.0):
                return jsonify({"error": "Sex value must be 0 or 1"}), 400
            input_values[1] = sex_float

        # Convert the rest of the inputs to numeric float values for prediction
        input_values = [float(i) if isinstance(i, (int, float)) else float(i) for i in input_values]

        # Predict using the heart disease model
        prediction = heart_model.predict([input_values])[0]

        # Return the result as JSON
        return jsonify({"result": "The Person has Heart Disease" if prediction == 1 else "The Person does not have Heart Disease"})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/predict/parkinsons', methods=['POST'])
def predict_parkinsons():
    data = request.get_json()
    input_values = np.array(data['input']).reshape(1, -1)
    prediction = parkinsons_model.predict(input_values)[0]
    result = "Parkinson's Detected" if prediction == 1 else "No Parkinson's"
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)