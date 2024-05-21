import joblib
import pandas as pd
import logging
from preprocess import preprocess_data

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def make_predictions(input_csv, output_csv):
    logging.info(f"Starting predictions with data from {input_csv}")

    # Preprocess new lead data and get the preprocessed DataFrame
    try:
        preprocessed_data = preprocess_data(input_csv, return_data=True)
        logging.info("Data preprocessed successfully")
    except Exception as e:
        logging.error(f"Failed to preprocess data: {e}")
        return

    # Load the trained model and scaler
    try:
        model = joblib.load('models/lead_conversion_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        with open('models/training_columns.pkl', 'rb') as f:
            model_columns = joblib.load(f)
        logging.info("Model, scaler, and training columns loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load model, scaler, or training columns: {e}")
        return

    # Ensure the preprocessed_data has the same columns as training data
    try:
        preprocessed_data = pd.get_dummies(preprocessed_data, drop_first=True)
        preprocessed_data = preprocessed_data.reindex(columns=model_columns, fill_value=0)
        preprocessed_data = scaler.transform(preprocessed_data)
        logging.info("Data standardized and reindexed successfully")
    except Exception as e:
        logging.error(f"Failed to standardize or reindex data: {e}")
        return

    # Predict conversion probability
    try:
        predictions = model.predict_proba(preprocessed_data)[:, 1] 
        logging.info("Predictions made successfully")
    except Exception as e:
        logging.error(f"Failed to make predictions: {e}")
        return

    # Create a DataFrame from the preprocessed data
    try:
        preprocessed_data = pd.DataFrame(preprocessed_data, columns=model_columns)
        preprocessed_data['conversion_probability'] = predictions
        logging.info("DataFrame with predictions created successfully")
    except Exception as e:
        logging.error(f"Failed to create DataFrame with predictions: {e}")
        return

    # Read original new_leads csv
    try:
        original_data = pd.read_csv(input_csv)
        logging.info("Original data loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load original data: {e}")
        return

    # Merge original data with predictions
    try:
        merged_data = pd.concat([original_data, preprocessed_data['conversion_probability']], axis=1)
        merged_data.to_csv(output_csv, index=False)
        logging.info(f"Predictions saved to {output_csv}")
    except Exception as e:
        logging.error(f"Failed to merge data or save predictions: {e}")
        return

if __name__ == '__main__':
    input_csv = 'data/new_leads.csv'  
    output_csv = 'data/predicted_leads.csv'
    make_predictions(input_csv, output_csv)
