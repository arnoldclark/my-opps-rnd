import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

def evaluate_model(data_file):
    # Load the preprocessed data
    data = pd.read_csv(data_file)
    
    # Separate features and target variable
    X = data.drop('IsPromoted', axis=1)  # Adjust 'converted' to your target column
    y = data['IsPromoted']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Load the scaler and model
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('lead_conversion_model.pkl')
    
    # Standardize the data
    X_test = scaler.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Classification Report:')
    print(class_report)

if __name__ == "__main__":
    data_file = 'data/preprocessed_prospects.csv'
    evaluate_model(data_file)
