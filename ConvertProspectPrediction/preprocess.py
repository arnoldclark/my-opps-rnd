import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_data(input_file, output_file=None, return_data=False):
    logging.info(f"Starting preprocessing of data from {input_file}")
    
    # Load the data, specifying 'NULL' as additional missing value
    try:
        data = pd.read_csv(input_file, na_values=['NULL'])
        logging.info("Data loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        return None

    # Fill missing values with 0
    data.fillna(0, inplace=True)
    logging.info("Missing values filled with 0")

    # Convert categorical variables to dummy variables
    data = pd.get_dummies(data, drop_first=True)
    logging.info("Categorical variables converted to dummy variables")

    # Save the preprocessed data if output_file is provided
    if output_file:
        try:
            data.to_csv(output_file, index=False)
            logging.info(f"Preprocessed data saved to {output_file}")
        except Exception as e:
            logging.error(f"Failed to save preprocessed data: {e}")
            return None

    # Return the preprocessed data if return_data is True
    if return_data:
        logging.info("Returning preprocessed data")
        return data

if __name__ == "__main__":
    input_file = 'data/production_training_data.csv'
    output_file = 'data/preprocessed_prospects.csv'
    
    # Save the preprocessed data to a file
    preprocess_data(input_file, output_file)
