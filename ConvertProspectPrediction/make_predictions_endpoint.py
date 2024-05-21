import joblib
import pandas as pd
import logging
from flask import Flask, request, jsonify
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

def preprocess_data(input_data):
    logging.info("Starting preprocessing of data")

    try:
        # Load data from JSON object
        data = pd.DataFrame(input_data)
        logging.info("Data loaded successfully from JSON")
    except Exception as e:
        logging.error(f"Failed to load data from JSON: {e}")
        return None

    # Preprocess the data
    try:
        # Replace NULL and empty values with 0
        data.replace(to_replace=[None, 'NULL', ''], value=0, inplace=True)
        logging.info("Missing values replaced with 0")

        # Convert categorical variables to dummy variables
        data = pd.get_dummies(data, drop_first=True)
        logging.info("Categorical variables converted to dummy variables")
    except Exception as e:
        logging.error(f"Failed to preprocess data: {e}")
        return None

    return data

def make_predictions(input_data):
    logging.info("Starting predictions")

    try:
        # Ensure input_data is not None
        if input_data is None:
            raise ValueError("Input data is None")

        # Preprocess new lead data and get the preprocessed DataFrame
        preprocessed_data = preprocess_data(input_data)
        logging.info("Data preprocessed successfully")

        # Load the trained model and scaler
        model = joblib.load('models/lead_conversion_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        with open('models/training_columns.pkl', 'rb') as f:
            model_columns = joblib.load(f)
        logging.info("Model, scaler, and training columns loaded successfully")

        # Ensure the preprocessed_data has the same columns as training data
        preprocessed_data = preprocessed_data.reindex(columns=model_columns, fill_value=0)
        preprocessed_data = scaler.transform(preprocessed_data)
        logging.info("Data standardized and reindexed successfully")

        # Predict conversion probability
        predictions = model.predict_proba(preprocessed_data)[:, 1]
        logging.info("Predictions made successfully")
        return predictions, None

    except Exception as e:
        logging.error(f"Failed to make predictions: {e}")
        return None, str(e)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Check if data is a list
        if isinstance(data, list):
            predictions = []
            errors = []
            
            # Iterate over each object in the array
            for lead_data in data:
                print("Lead data:", lead_data)  # Print lead data for debugging purposes
                input_data = pd.DataFrame([lead_data])  # Convert each object to a DataFrame
                
                # Make predictions for the current lead data
                prediction, error = make_predictions(input_data)
                if prediction is not None:
                    predictions.append(prediction.tolist())
                    errors.append(None)
                else:
                    predictions.append(None)
                    errors.append(error)
            
            # Return predictions and errors for each lead data point
            return jsonify({'predictions': predictions, 'errors': errors})
        else:
            return jsonify({'error': 'Request body must be a JSON array of objects'}), 400
    
    except Exception as e:
        logging.error(f"Error in /predict endpoint: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
