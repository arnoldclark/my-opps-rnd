import pandas as pd

def preprocess_data(input_file, output_file=None, return_data=False):
    # Load the data, specifying 'NULL' as additional missing value
    data = pd.read_csv(input_file, na_values=['NULL'])
    
    # Fill missing values with 0
    data.fillna(0, inplace=True)
    
    # Convert categorical variables to dummy variables
    data = pd.get_dummies(data, drop_first=True)
    
    # Save the preprocessed data if output_file is provided
    if output_file:
        data.to_csv(output_file, index=False)
        print(f"Preprocessed data saved to {output_file}")
    
    # Return the preprocessed data if return_data is True
    if return_data:
        return data

if __name__ == "__main__":
    input_file = 'data/production_training_data.csv'
    output_file = 'data/preprocessed_prospects.csv'
    
    # Save the preprocessed data to a file
    preprocess_data(input_file, output_file)
