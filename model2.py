import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             precision_recall_curve, auc, f1_score, precision_score, recall_score)
# import matplotlib.pyplot as plt  # Optional for local runs unless saving plots
# import seaborn as sns            # Optional for local runs unless saving plots
import xgboost as xgb
import sys
import joblib
import time # To time the process

print("--- Manual XGBoost Hyperparameter Tuning (Local Simulation) ---")

# --- Configuration ---
TRAINING_DATA_FILE = 'training_data_unscaled.csv'
BEST_MODEL_SAVE_FILE = 'creditguard_xgb_best_tuned.joblib' # Save the best model found
TEST_SET_SIZE = 0.2 # For final evaluation split
VALIDATION_SET_SIZE = 0.2 # Split from training data for tuning evaluation
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

X_full = data.drop(target_col, axis=1)
y_full = data[target_col]
if not np.issubdtype(y_full.dtype, np.integer): y_full = y_full.astype(int)

print(f"\nFull Features shape: {X_full.shape}"); print(f"Full Target shape: {y_full.shape}")
print(f"Class distribution in full data:\n{y_full.value_counts(normalize=True)}")

# --- 3. Train / Test Split (for FINAL evaluation AFTER tuning) ---
# We split off a final test set *first* and don't touch it during tuning.
print("\n--- Splitting Data into Train+Validation / Test ---")
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_full, y_full, test_size=TEST_SET_SIZE, random_state=RANDOM_STATE, stratify=y_full
)
print(f"Train+Validation set shape: {X_train_val.shape}")
print(f"Test set shape: {X_test.shape}")


# --- 4. Define Hyperparameter Sets to Try ---
# This is where you define the different combinations SageMaker would explore.
# Add more combinations to explore a wider space.
hyperparameter_sets = [
    { # Set 1: Baseline from original script
        'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 5,
        'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0,
        'early_stopping_rounds': 20
    },
    { # Set 2: Deeper trees, lower learning rate
        'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 7,
        'subsample': 0.7, 'colsample_bytree': 0.7, 'gamma': 0.1,
        'early_stopping_rounds': 25
    },
    { # Set 3: Shallower trees, more regularization
        'n_estimators': 250, 'learning_rate': 0.1, 'max_depth': 4,
        'subsample': 0.9, 'colsample_bytree': 0.9, 'gamma': 0.2,
        'early_stopping_rounds': 20
    },
    { # Set 4: Aggressive subsampling
        'n_estimators': 300, 'learning_rate': 0.08, 'max_depth': 6,
        'subsample': 0.6, 'colsample_bytree': 0.6, 'gamma': 0.05,
        'early_stopping_rounds': 25
    },
    # Add more sets here to explore...
]

# --- 5. Tuning Loop ---
results = []
best_score = -1 # Initialize with a value lower than any possible score
best_hyperparams = None
best_model = None
best_iteration_count = None

print(f"\n--- Starting Manual Hyperparameter Search ({len(hyperparameter_sets)} sets) ---")
start_time = time.time()

# Split Train+Validation into Train and Validation sets FOR EACH RUN?
# No, keep it consistent for fair comparison across hyperparams.
# Split ONCE here before the loop.
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=VALIDATION_SET_SIZE, random_state=RANDOM_STATE, stratify=y_train_val
)
print(f"Actual Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")

# Calculate scale_pos_weight based on the ACTUAL training set
neg_count, pos_count = np.bincount(y_train)
if pos_count == 0:
    print("ERROR: No positive samples (fraud) in the training set used for tuning!")
    scale_pos_weight_value = 1
else:
    scale_pos_weight_value = neg_count / pos_count
print(f'Calculated scale_pos_weight for tuning: {scale_pos_weight_value:.2f}')

for i, params in enumerate(hyperparameter_sets):
    print(f"\n--- Training Run {i+1}/{len(hyperparameter_sets)} ---")
    print(f"Hyperparameters: {params}")

    try:
        # Initialize XGBoost Classifier with current hyperparameters
        xgb_model_tune = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='aucpr', # Optimize for AUC-PR on validation set
            scale_pos_weight=scale_pos_weight_value,
            use_label_encoder=False,
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            max_depth=params['max_depth'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            gamma=params['gamma'],
            early_stopping_rounds=params['early_stopping_rounds'],
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        # Use the VALIDATION set for early stopping and evaluation metric
        eval_set = [(X_val, y_val)]

        print("Training...")
        xgb_model_tune.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False # Keep logs clean during tuning loop
        )
        print("Training complete.")

        # Evaluate on the VALIDATION set to get the score for tuning
        y_pred_proba_val = xgb_model_tune.predict_proba(X_val)[:, 1]
        # val_auc_roc = roc_auc_score(y_val, y_pred_proba_val) # Alternative metric
        pr_precision_val, pr_recall_val, _ = precision_recall_curve(y_val, y_pred_proba_val)
        val_auc_pr = auc(pr_recall_val, pr_precision_val) # Our target metric

        iteration_count = xgb_model_tune.best_iteration if hasattr(xgb_model_tune, 'best_iteration') else params['n_estimators']
        print(f"Validation AUC-PR: {val_auc_pr:.5f} (at iteration {iteration_count})")

        results.append({'params': params, 'validation_auc_pr': val_auc_pr, 'iterations': iteration_count})

        # Check if this is the best score so far
        if val_auc_pr > best_score:
            print(f"*** New Best Score Found! Previous best: {best_score:.5f} ***")
            best_score = val_auc_pr
            best_hyperparams = params
            best_model = xgb_model_tune # Keep the actual trained model object
            best_iteration_count = iteration_count

    except Exception as e:
        print(f"ERROR during training run {i+1} with params {params}: {e}")
        results.append({'params': params, 'validation_auc_pr': -1, 'iterations': -1, 'error': str(e)}) # Log error

total_time = time.time() - start_time
print(f"\n--- Manual Tuning Finished ---")
print(f"Total time: {total_time:.2f} seconds")

# --- 6. Report Best Results ---
print("\n--- Best Hyperparameters Found ---")
if best_hyperparams:
    print(f"Best Validation AUC-PR: {best_score:.5f}")
    print(f"Achieved at iteration: {best_iteration_count}")
    print("Best Hyperparameters:")
    for key, value in best_hyperparams.items():
        print(f"  {key}: {value}")

    # --- 7. Evaluate the BEST Model on the Held-Out TEST Set ---
    print("\n--- Evaluating BEST Model on Test Set ---")
    y_pred_proba_test = best_model.predict_proba(X_test)[:, 1]
    y_pred_class_test = best_model.predict(X_test)

    print("\nConfusion Matrix (Test Set - Best Model):")
    print(confusion_matrix(y_test, y_pred_class_test))

    print("\nClassification Report (Test Set - Best Model):")
    print(classification_report(y_test, y_pred_class_test, target_names=['Non-Fraud (0)', 'Fraud (1)']))

    precision_test = precision_score(y_test, y_pred_class_test)
    recall_test = recall_score(y_test, y_pred_class_test)
    f1_test = f1_score(y_test, y_pred_class_test)
    auc_roc_test = roc_auc_score(y_test, y_pred_proba_test)
    pr_precision_test, pr_recall_test, _ = precision_recall_curve(y_test, y_pred_proba_test)
    auc_pr_test = auc(pr_recall_test, pr_precision_test)

    print("\nKey Test Set Metrics (Best Model):")
    print(f"Precision: {precision_test:.4f}")
    print(f"Recall:    {recall_test:.4f}")
    print(f"F1-Score:  {f1_test:.4f}")
    print(f"AUC-ROC:   {auc_roc_test:.4f}")
    print(f"AUC-PR:    {auc_pr_test:.4f}")

    # --- 8. Save the BEST Trained Model ---
    print(f"\n--- Saving BEST Trained XGBoost Model to {BEST_MODEL_SAVE_FILE} ---")
    try:
        joblib.dump(best_model, BEST_MODEL_SAVE_FILE)
        print("Best model saved successfully.")
    except Exception as e:
        print(f"ERROR: Could not save best model: {e}")

else:
    print("No successful training runs completed. Could not determine best model.")

print("\n--- Local Tuning Simulation Finished ---")

# Optional: Print all results for comparison
# print("\n--- All Tuning Results ---")
# for res in sorted(results, key=lambda x: x['validation_auc_pr'], reverse=True):
#     print(f"Score: {res['validation_auc_pr']:.5f}, Iterations: {res['iterations']}, Params: {res['params']}")