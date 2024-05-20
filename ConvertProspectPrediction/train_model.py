import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def train_model(data_file):
    # Load the preprocessed data
    data = pd.read_csv(data_file)

    # Separate features and target variable
    X = data.drop('IsPromoted', axis=1)
    y = data['IsPromoted']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Progress bar
    epochs = 100
    with tqdm(total=epochs, desc="Training", unit="epoch") as pbar:
        for _ in range(epochs):
            model.fit(X_train_scaled, y_train)
            pbar.update(1)

    # Save the model columns
    model_columns = X.columns
    joblib.dump(model_columns, 'training_columns.pkl')

    # Save the model and scaler
    joblib.dump(model, 'lead_conversion_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    # Confusion Matrix
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # ROC Curve
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
    train_model(data_file)
