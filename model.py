import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.utils import class_weight # Not directly used in XGBoost in the same way
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             precision_recall_curve, auc, f1_score, precision_score, recall_score)
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb # Import XGBoost
import sys
import joblib # To save the model

print("--- Phase 2: XGBoost Fraud Detection Model Training on Synthetic Data ---")

# --- Configuration ---
TRAINING_DATA_FILE = 'training_data_unscaled.csv' # Input from Phase 1
MODEL_SAVE_FILE = 'creditguard_xgb.joblib' # Output XGBoost model file
TEST_SET_SIZE = 0.2
RANDOM_STATE = 42

# --- 1. Load Processed Data ---
try:
    data = pd.read_csv(TRAINING_DATA_FILE)
    print(f"Loaded processed data from '{TRAINING_DATA_FILE}'. Shape: {data.shape}")
    if data.empty: sys.exit("Error: Loaded data is empty. Exiting.")
except FileNotFoundError:
    print(f"ERROR: Training data file '{TRAINING_DATA_FILE}' not found.")
    sys.exit("Please run the preprocessing script first.")
except Exception as e:
    print(f"Error loading data: {e}"); sys.exit(1)

# --- 2. Define Features (X) and Target (y) ---
target_col = data.columns[-1]
if target_col.lower() != 'isfraud':
    print(f"Warning: Assuming last column '{target_col}' is the target.")

X = data.drop(target_col, axis=1)
y = data[target_col]
if not np.issubdtype(y.dtype, np.integer): y = y.astype(int)

print(f"\nFeatures shape: {X.shape}"); print(f"Target shape: {y.shape}")
print("Input features:", X.columns.tolist())
print(f"Class distribution in loaded data:\n{y.value_counts(normalize=True)}")

# --- 3. Train/Test Split (Stratified) ---
print("\n--- Splitting Data ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SET_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"Training set shape: {X_train.shape}, Target: {y_train.shape}")
print(f"Testing set shape: {X_test.shape}, Target: {y_test.shape}")
print(f"Fraud cases in training set: {sum(y_train)}")
print(f"Fraud cases in testing set: {sum(y_test)}")

# --- 4. Handling Imbalance - Calculate 'scale_pos_weight' for XGBoost ---
# XGBoost uses a specific parameter `scale_pos_weight` for imbalance.
# It's typically calculated as: count(negative_class) / count(positive_class)
print("\n--- Calculating scale_pos_weight for XGBoost ---")
neg_count, pos_count = np.bincount(y_train)
if pos_count == 0:
    print("ERROR: No positive samples (fraud) in the training set!")
    scale_pos_weight_value = 1 # Default if no positive samples
else:
    scale_pos_weight_value = neg_count / pos_count
print(f'Training samples: Total={neg_count+pos_count}, Non-Fraud={neg_count}, Fraud={pos_count}')
print(f'Calculated scale_pos_weight: {scale_pos_weight_value:.2f}')
print("(XGBoost will give fraud cases this much more weight)")

# --- 5. Define and Train XGBoost Model ---

# Initialize XGBoost Classifier
# Common parameters:
# n_estimators: number of boosting rounds (trees)
# max_depth: max depth of each tree
# learning_rate: step size shrinkage
# objective: 'binary:logistic' for binary classification (outputs probability)
# scale_pos_weight: handles imbalance
# use_label_encoder=False: recommended nowadays
# eval_metric: metric for early stopping (e.g., 'aucpr' for Area Under PR Curve)
# early_stopping_rounds: stops training if eval metric doesn't improve
print("\n--- Defining and Training XGBoost Model ---")

# Initialize XGBoost Classifier WITH early stopping parameter
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='aucpr',          # Evaluate using Area Under PR Curve
    scale_pos_weight=scale_pos_weight_value, # Handle imbalance
    use_label_encoder=False,    # Recommended setting
    n_estimators=200,           # Start with a reasonable number of trees
    learning_rate=0.1,          # Common learning rate
    max_depth=5,                # Controls tree complexity
    subsample=0.8,              # Fraction of samples used per tree
    colsample_bytree=0.8,       # Fraction of features used per tree
    gamma=0,                    # Minimum loss reduction for split (regularization)
    early_stopping_rounds=20,   # <<< MOVED HERE! Stop if AUC-PR on test set doesn't improve
    random_state=RANDOM_STATE,
    n_jobs=-1                   # Use all available CPU cores
)

# XGBoost still needs an evaluation set to PERFORM the early stopping
eval_set = [(X_test, y_test)] # Evaluate directly on the test set during training

print("Training...")
# Fit the model, passing only the necessary arguments
xgb_model.fit(
    X_train,
    y_train,
    eval_set=eval_set,
    verbose=False             # Set to True or a number to see progress
)
print("Training complete.")
if hasattr(xgb_model, 'best_iteration'): # Check if early stopping occurred
    print(f"Best iteration (tree): {xgb_model.best_iteration}")
else:
    print("Early stopping did not occur (trained for full n_estimators).")
print(f"Best iteration (tree): {xgb_model.best_iteration}") # Check where it stopped

# --- 6. Evaluate the Model on the TEST Set ---
print("\n--- Evaluating Model on Test Set ---")
y_pred_proba_test = xgb_model.predict_proba(X_test)[:, 1] # Probability of class 1
y_pred_class_test = xgb_model.predict(X_test)          # Class prediction (uses 0.5 threshold by default)

print("\nConfusion Matrix (Test Set):")
print(confusion_matrix(y_test, y_pred_class_test))

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred_class_test, target_names=['Non-Fraud (0)', 'Fraud (1)']))

# Calculate key metrics explicitly
precision = precision_score(y_test, y_pred_class_test)
recall = recall_score(y_test, y_pred_class_test)
f1 = f1_score(y_test, y_pred_class_test)
auc_roc = roc_auc_score(y_test, y_pred_proba_test)
pr_precision, pr_recall, _ = precision_recall_curve(y_test, y_pred_proba_test)
auc_pr = auc(pr_recall, pr_precision)

print("\nKey Test Set Metrics:")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"AUC-ROC:   {auc_roc:.4f}")
print(f"AUC-PR:    {auc_pr:.4f}")

# --- 7. Feature Importance (Optional but insightful) ---
print("\n--- Feature Importance ---")
try:
    importance = xgb_model.get_booster().get_score(importance_type='weight') # or 'gain' or 'cover'
    feature_importance = pd.DataFrame({
        'Feature': list(importance.keys()),
        'Importance': list(importance.values())
    }).sort_values('Importance', ascending=False)

    print(feature_importance)

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10)) # Plot top 10
    plt.title('XGBoost Feature Importance (Weight)')
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Could not generate feature importance plot: {e}")


# --- 8. Save the Trained Model ---
print(f"\n--- Saving Trained XGBoost Model to {MODEL_SAVE_FILE} ---")
try:
    joblib.dump(xgb_model, MODEL_SAVE_FILE)
    print("Model saved successfully using joblib.")
    # Alternatively, XGBoost has its own save method:
    # xgb_model.save_model(MODEL_SAVE_FILE.replace('.joblib', '.xgb'))
    # print("Model saved successfully using XGBoost format.")
except Exception as e:
    print(f"ERROR: Could not save model: {e}")

print("\n--- Phase 2 Model Training Finished ---")