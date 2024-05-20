import joblib
import pandas as pd
from preprocess import preprocess_data  # Importing the preprocess_data function from preprocess.py

def make_predictions(input_csv, output_csv):
    # Preprocess new lead data and get the preprocessed DataFrame
    preprocessed_data = preprocess_data(input_csv, return_data=True)

    # Load the trained model and scaler
    model = joblib.load('lead_conversion_model.pkl')
    scaler = joblib.load('scaler.pkl')
    with open('training_columns.pkl', 'rb') as f:
        model_columns = joblib.load(f)

    # Ensure the preprocessed_data has the same columns as training data
    preprocessed_data = pd.get_dummies(preprocessed_data, drop_first=True)
    preprocessed_data = preprocessed_data.reindex(columns=model_columns, fill_value=0)
    preprocessed_data = scaler.transform(preprocessed_data)

    # Predict conversion probability
    predictions = model.predict_proba(preprocessed_data)[:, 1]  # Assuming binary classification, we take probability of class 1

    # Create a DataFrame from the preprocessed data
    preprocessed_data = pd.DataFrame(preprocessed_data, columns=model_columns)
    
    # Add predictions to the DataFrame
    preprocessed_data['conversion_probability'] = predictions

    # Read original new_leads csv
    original_data = pd.read_csv(input_csv)

    # Merge original data with predictions
    merged_data = pd.concat([original_data, preprocessed_data['conversion_probability']], axis=1)

    # Save the merged DataFrame to a new CSV file
    merged_data.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

if __name__ == '__main__':
    input_csv = 'data/new_leads.csv'  # Replace with your input CSV file
    output_csv = 'data/predicted_leads.csv'  # Replace with your desired output CSV file
    make_predictions(input_csv, output_csv)
