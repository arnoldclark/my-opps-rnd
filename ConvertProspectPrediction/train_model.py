import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(data_file):
    logging.info("Loading the preprocessed data...")
    data = pd.read_csv(data_file)

    logging.info("Separating features and target variable...")
    X = data.drop('IsPromoted', axis=1)
    y = data['IsPromoted']
    
    logging.info("Splitting the data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    logging.info("Standardizing the data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logging.info("Initializing and training the model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    epochs = 5
    with tqdm(total=epochs, desc="Training", unit="epoch") as pbar:
        for _ in range(epochs):
            model.fit(X_train_scaled, y_train)
            pbar.update(1)

    logging.info("Saving the model columns...")
    model_columns = X.columns
    joblib.dump(model_columns, 'models/training_columns.pkl')

    logging.info("Saving the model and scaler...")
    joblib.dump(model, 'models/lead_conversion_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')

    logging.info("Generating the Confusion Matrix...")
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    logging.info("Generating the ROC Curve...")
    y_score = model.predict_proba(X_test_scaled)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    data_file = 'data/preprocessed_prospects.csv'
    logging.info(f"Starting the training process with data file: {data_file}")
    train_model(data_file)
    logging.info("Training process completed.")
