import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model(data_file):
    logging.info(f"Starting model evaluation with data from {data_file}")
    
    # Load the preprocessed data
    try:
        data = pd.read_csv(data_file)
        logging.info("Data loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        return

    # Separate features and target variable
    try:
        X = data.drop('IsPromoted', axis=1)  # Adjust 'IsPromoted' to your target column
        y = data['IsPromoted']
        logging.info("Separated features and target variable")
    except Exception as e:
        logging.error(f"Failed to separate features and target variable: {e}")
        return
    
    # Split the data
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        logging.info("Data split into training and test sets")
    except Exception as e:
        logging.error(f"Failed to split data: {e}")
        return

    # Load the scaler and model
    try:
        scaler = joblib.load('models/scaler.pkl')
        model = joblib.load('models/lead_conversion_model.pkl')
        logging.info("Scaler and model loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load scaler or model: {e}")
        return

    # Standardize the data
    try:
        X_test = scaler.transform(X_test)
        logging.info("Test data standardized")
    except Exception as e:
        logging.error(f"Failed to standardize test data: {e}")
        return

    # Make predictions
    try:
        y_pred = model.predict(X_test)
        logging.info("Predictions made on test data")
    except Exception as e:
        logging.error(f"Failed to make predictions: {e}")
        return

    # Evaluate the model
    try:
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        
        logging.info(f'Accuracy: {accuracy * 100:.2f}%')
        logging.info('Confusion Matrix:\n' + str(conf_matrix))
        logging.info('Classification Report:\n' + class_report)
    except Exception as e:
        logging.error(f"Failed to evaluate model: {e}")
        return

if __name__ == "__main__":
    data_file = 'data/preprocessed_prospects.csv'
    evaluate_model(data_file)
