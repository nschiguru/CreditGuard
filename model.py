import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler # RobustScaler less sensitive to outliers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
# For SMOTE (if chosen for imbalance):
# from imblearn.over_sampling import SMOTE
# from imblearn.pipeline import Pipeline # Helpful for combining steps

# --- 1. Load Data ---
try:
    data = pd.read_csv('creditcard.csv') # Make sure this file is downloaded and accessible
    print("Data loaded successfully.")
    print(data.head())
except FileNotFoundError:
    print("Error: creditcard.csv not found. Please download it.")
    exit()

# --- 2. Preprocessing ---
print("Preprocessing data...")

# Drop 'Time' if not using it for sequence (often not useful in this static form)
# data = data.drop('Time', axis=1)

# Scale 'Amount' and 'Time' (if kept)
# scaler = StandardScaler()
scaler = RobustScaler() # Often better for data with potential outliers like Amount

# Scale 'Amount' - Create a copy first to avoid SettingWithCopyWarning
data_scaled = data.copy()
data_scaled['scaled_Amount'] = scaler.fit_transform(data_scaled['Amount'].values.reshape(-1, 1))
# data_scaled['scaled_Time'] = scaler.fit_transform(data_scaled['Time'].values.reshape(-1, 1)) # if keeping Time

# Drop original columns
data_scaled = data_scaled.drop(['Amount', 'Time'], axis=1, errors='ignore') # Ignore error if Time was already dropped

# Define Features (X) and Target (y)
X = data_scaled.drop('Class', axis=1)
y = data_scaled['Class']

# --- 3. Train/Test Split (Stratified) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
print(f"Fraud cases in training set: {sum(y_train)}")
print(f"Fraud cases in testing set: {sum(y_test)}")


# --- 4. Handle Imbalance (Example: using class_weight) ---
# Option A: Using class_weight parameter (simple for some models)
# model = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1) # Use balanced RF

# Option B: Using SMOTE (Requires imblearn)
# print("Applying SMOTE...")
# smote = SMOTE(random_state=42)
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
# print(f"Resampled training set shape: {X_train_resampled.shape}")
# print(f"Resampled fraud cases: {sum(y_train_resampled)}")
# model = LogisticRegression(solver='liblinear', random_state=42) # Train on resampled data
# model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) # Train on resampled data

# --- 5. Train Model ---
print(f"Training {model.__class__.__name__}...")
# If using SMOTE, train on resampled data:
# model.fit(X_train_resampled, y_train_resampled)
# If using class_weight, train on original stratified data:
model.fit(X_train, y_train)
print("Training complete.")

# --- 6. Evaluate Model ---
print("Evaluating model on the test set...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1] # Probability of class 1 (fraud)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Non-Fraud', 'Fraud']))

print(f"\nArea Under ROC Curve (AUC-ROC): {roc_auc_score(y_test, y_pred_proba):.4f}")

# Calculate Precision-Recall Curve and AUC-PR
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
auc_pr = auc(recall, precision)
print(f"Area Under Precision-Recall Curve (AUC-PR): {auc_pr:.4f}")


# --- Next Steps ---
# - Save the trained model (e.g., using joblib or pickle)
# - Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
# - Try different models (XGBoost, LightGBM, Neural Networks)
# - If using AWS, deploy the saved model to a SageMaker Endpoint
# - Modify the Lambda function to call the SageMaker Endpoint