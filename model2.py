import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             precision_recall_curve, auc, f1_score, precision_score, recall_score,
                             fbeta_score) # Import fbeta_score
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import sys
import joblib
import time
import json
import os

print("--- Local XGBoost Hyperparameter Tuning (Simulating SageMaker) ---")
print("--- Goal: Achieve >= 90% Precision on Test Set via Threshold Tuning ---")

# --- Configuration ---
TRAINING_DATA_FILE = 'training_data_unscaled.csv'
BEST_MODEL_SAVE_FILE = 'creditguard_xgb_best_local_tuned.joblib'
THRESHOLD_SAVE_FILE = 'optimal_threshold_local.json' # Save the threshold too
FINAL_EVALUATION_FILE = 'final_evaluation_local.json'

TARGET_PRECISION = 0.90 # <<< YOUR PRECISION TARGET
RANDOM_STATE = 42

# Data Split Ratios
TEST_SET_SIZE = 0.20       # Reserve 20% for final unbiased test
VALIDATION_SET_SIZE = 0.25 # Use 25% OF THE REMAINING for validation/thresholding (i.e., 0.25 * 0.80 = 20% of total)

# --- Helper Function for Metrics ---
def log_metrics(y_true, y_pred_proba, y_pred_class, prefix="test", threshold=0.5):
    """Calculates, prints, and returns metrics for logging."""
    metrics_dict = {}
    try:
        precision = precision_score(y_true, y_pred_class, zero_division=0)
        recall = recall_score(y_true, y_pred_class, zero_division=0)
        f1 = f1_score(y_true, y_pred_class, zero_division=0)
        fbeta_0_5 = fbeta_score(y_true, y_pred_class, beta=0.5, zero_division=0) # F0.5 score
        auc_roc = -1.0
        auc_pr = -1.0

        # AUC requires probabilities and both classes present
        if len(np.unique(y_true)) > 1:
            auc_roc = roc_auc_score(y_true, y_pred_proba)
            pr_precision_curve, pr_recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
            auc_pr = auc(pr_recall_curve, pr_precision_curve)
        else:
            print(f"WARN: Only one class present in {prefix} y_true. Cannot calculate AUCs.")


        print(f"\n--- Evaluation Metrics ({prefix} set, Threshold: {threshold:.4f}) ---")
        print(f"Precision (Target >= {TARGET_PRECISION:.2f}): {precision:.4f}")
        print(f"Recall:                    {recall:.4f}")
        print(f"F1-Score:                  {f1:.4f}")
        print(f"F0.5-Score:                {fbeta_0_5:.4f}") # Report F0.5
        print(f"AUC-ROC:                   {auc_roc:.4f}")
        print(f"AUC-PR:                    {auc_pr:.4f}")

        print(f"\nConfusion Matrix ({prefix} Set, Threshold: {threshold:.4f}):")
        cm = confusion_matrix(y_true, y_pred_class)
        print(cm)

        print(f"\nClassification Report ({prefix} Set, Threshold: {threshold:.4f}):")
        print(classification_report(y_true, y_pred_class, target_names=['Non-Fraud (0)', 'Fraud (1)'], zero_division=0))

        metrics_dict = {
            f'{prefix}_precision': float(precision),
            f'{prefix}_recall': float(recall),
            f'{prefix}_f1': float(f1),
            f'{prefix}_f0.5': float(fbeta_0_5),
            f'{prefix}_auc_roc': float(auc_roc),
            f'{prefix}_auc_pr': float(auc_pr),
            f'{prefix}_threshold': float(threshold),
            # Add confusion matrix elements for potential later analysis
            f'{prefix}_cm_tn': int(cm[0, 0]) if cm.shape == (2, 2) else -1,
            f'{prefix}_cm_fp': int(cm[0, 1]) if cm.shape == (2, 2) else -1,
            f'{prefix}_cm_fn': int(cm[1, 0]) if cm.shape == (2, 2) else -1,
            f'{prefix}_cm_tp': int(cm[1, 1]) if cm.shape == (2, 2) else -1,
        }
        return metrics_dict

    except Exception as e:
        print(f"Error calculating/logging {prefix} metrics: {e}")
        return {}

# --- Helper Function for Threshold Finding ---
def find_optimal_threshold(model, X_val, y_val, target_precision):
    """Finds threshold on validation set meeting target precision."""
    print(f"\n--- Finding Threshold for Target Precision >= {target_precision:.2f} on Validation Set ---")
    if len(np.unique(y_val)) < 2:
        print("WARN: Validation set has only one class. Cannot calculate PR curve. Using default 0.5 threshold.")
        return 0.5
    if sum(y_val) == 0:
         print("WARN: Validation set has no positive samples. Cannot tune threshold effectively. Using default 0.5 threshold.")
         return 0.5

    y_pred_proba_val = model.predict_proba(X_val)[:, 1]
    precision_val, recall_val, thresholds_val = precision_recall_curve(y_val, y_pred_proba_val)

    optimal_threshold = 0.5 # Default fallback
    found_threshold = False

    # Iterate thresholds (associated with recall/precision pairs) to find first meeting criteria
    # precision_val[i] corresponds to thresholds_val[i]
    valid_indices = [i for i, p in enumerate(precision_val) if p >= target_precision and i < len(thresholds_val)]

    if valid_indices:
        # Choose the index that gives the highest recall among those meeting the precision target
        best_val_idx_for_prec = -1
        max_recall_at_prec = -1
        for i in valid_indices:
            if recall_val[i] > max_recall_at_prec:
                 max_recall_at_prec = recall_val[i]
                 best_val_idx_for_prec = i

        optimal_threshold = thresholds_val[best_val_idx_for_prec]
        print(f"Found threshold: {optimal_threshold:.4f} -> (Val Precision={precision_val[best_val_idx_for_prec]:.4f}, Val Recall={recall_val[best_val_idx_for_prec]:.4f})")
        found_threshold = True

    # Check the last point (highest precision, potentially lowest recall) if needed
    elif precision_val[-1] >= target_precision:
         optimal_threshold = thresholds_val[-1] if thresholds_val.size > 0 else 1.0 # Handle edge case
         print(f"Target precision met only near highest confidence. Using threshold: {optimal_threshold:.4f} (Val Precision={precision_val[-1]:.4f}, Val Recall={recall_val[-1]:.4f})")
         found_threshold = True

    if not found_threshold:
        # Find the threshold that gives the highest possible precision if target cannot be met
        max_prec_idx = np.argmax(precision_val[:-1]) # Exclude last point which might have recall 0
        optimal_threshold = thresholds_val[max_prec_idx]
        print(f"WARN: Could not meet target precision {target_precision:.2f} on validation set.")
        print(f"      Highest achieved precision: {precision_val[max_prec_idx]:.4f} at threshold {optimal_threshold:.4f}.")
        print(f"      Using threshold {optimal_threshold:.4f} which yields the max validation precision.")
        # Or uncomment below to fallback to 0.5 instead
        # print(f"      Falling back to default threshold 0.5")
        # optimal_threshold = 0.5

    return optimal_threshold

# --- 1. Load Data ---
print(f"\nLoading data from: {TRAINING_DATA_FILE}")
try:
    data = pd.read_csv(TRAINING_DATA_FILE)
    print(f"Loaded data shape: {data.shape}")
    if data.empty: sys.exit("Error: Loaded data is empty.")
except Exception as e:
    print(f"ERROR loading data: {e}"); sys.exit(1)

# --- 2. Define Features/Target & Initial Split (Train+Val / Test) ---
target_col = data.columns[-1]
if target_col.lower() != 'isfraud':
    print(f"Warning: Assuming last column '{target_col}' is the target.")
X_full = data.drop(target_col, axis=1)
y_full = data[target_col].astype(int)
print(f"Full data class distribution:\n{y_full.value_counts(normalize=True)}")

# Split off final test set FIRST
print(f"\nSplitting off Test set ({TEST_SET_SIZE*100:.0f}%)...")
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_full, y_full, test_size=TEST_SET_SIZE, random_state=RANDOM_STATE, stratify=y_full
)

# Split remaining data into Train and Validation
print(f"Splitting remaining into Train/Validation ({VALIDATION_SET_SIZE*100:.0f}% of rest)...")
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=VALIDATION_SET_SIZE, random_state=RANDOM_STATE, stratify=y_train_val
)
print(f"\nData Split Shapes:")
print(f"  Training set:   {X_train.shape} (Fraud cases: {sum(y_train)})")
print(f"  Validation set: {X_val.shape} (Fraud cases: {sum(y_val)})")
print(f"  Test set:       {X_test.shape} (Fraud cases: {sum(y_test)})")

# --- 3. Calculate scale_pos_weight (on Train set ONLY) ---
print("\nCalculating scale_pos_weight on training data...")
neg_count, pos_count = np.bincount(y_train)
if pos_count == 0:
    print("ERROR: No positive samples in the final training set!")
    scale_pos_weight_value = 1
else:
    scale_pos_weight_value = neg_count / pos_count
print(f'Calculated scale_pos_weight: {scale_pos_weight_value:.2f}')

# --- 4. Define Hyperparameter Sets to Try ---
# Add more diverse sets to explore the space better
# Focus parameters that might influence precision/recall trade-off (e.g., regularization, depth)
hyperparameter_sets = [
    { 'n_estimators': 150, 'learning_rate': 0.1, 'max_depth': 4, 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.1, 'early_stopping_rounds': 20},
    { 'n_estimators': 250, 'learning_rate': 0.05, 'max_depth': 6, 'subsample': 0.7, 'colsample_bytree': 0.7, 'gamma': 0.2, 'early_stopping_rounds': 25},
    { 'n_estimators': 300, 'learning_rate': 0.08, 'max_depth': 5, 'subsample': 0.9, 'colsample_bytree': 0.9, 'gamma': 0.05, 'early_stopping_rounds': 25},
    { 'n_estimators': 200, 'learning_rate': 0.15, 'max_depth': 3, 'subsample': 0.75, 'colsample_bytree': 0.75, 'gamma': 0.3, 'early_stopping_rounds': 20},
    { 'n_estimators': 350, 'learning_rate': 0.03, 'max_depth': 7, 'subsample': 0.65, 'colsample_bytree': 0.65, 'gamma': 0.15, 'early_stopping_rounds': 30},
]

# --- 5. Tuning Loop ---
results = []
best_objective_score = -1 # Track highest F0.5 score on validation set
best_hyperparams = None
best_model = None
best_iteration_count = None

print(f"\n--- Starting Local Hyperparameter Search ({len(hyperparameter_sets)} sets) ---")
print(f"--- Objective: Maximize F0.5 Score on Validation Set (Threshold 0.5) ---")
start_time = time.time()

for i, params in enumerate(hyperparameter_sets):
    print(f"\n--- Training Run {i+1}/{len(hyperparameter_sets)} ---")
    print(f"Hyperparameters: {params}")
    run_start_time = time.time()

    try:
        # Initialize XGBoost Classifier with current hyperparameters
        xgb_model_tune = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='aucpr', # Use AUC-PR for early stopping guidance
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

        # Use the VALIDATION set for early stopping
        eval_set = [(X_val, y_val)]

        print("Training...")
        xgb_model_tune.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False # Keep logs cleaner during loop
        )
        iter_count = xgb_model_tune.best_iteration if hasattr(xgb_model_tune, 'best_iteration') else params['n_estimators']
        print(f"Training complete (iterations: {iter_count}).")

        # --- Evaluate on VALIDATION set to get the score for tuning ---
        # We use F0.5 score at default 0.5 threshold here to select the best candidate model
        y_pred_proba_val = xgb_model_tune.predict_proba(X_val)[:, 1]
        y_pred_class_val_05 = (y_pred_proba_val >= 0.5).astype(int) # Use 0.5 for F-beta objective calc

        if len(np.unique(y_val)) < 2:
             print("WARN: Only one class in validation set for this run. Cannot calculate F0.5. Skipping score update.")
             current_objective_score = -1 # Cannot score
        else:
             current_objective_score = fbeta_score(y_val, y_pred_class_val_05, beta=0.5, zero_division=0)

        print(f"Validation F0.5 Score (@0.5 thresh): {current_objective_score:.5f}")
        results.append({'params': params, 'validation_f0.5_score': current_objective_score, 'iterations': iter_count})

        # Check if this is the best score so far
        if current_objective_score > best_objective_score:
            print(f"*** New Best F0.5 Score Found! Previous best: {best_objective_score:.5f} ***")
            best_objective_score = current_objective_score
            best_hyperparams = params
            best_model = xgb_model_tune # Keep the actual trained model object
            best_iteration_count = iter_count

    except Exception as e:
        print(f"ERROR during training run {i+1} with params {params}: {e}")
        results.append({'params': params, 'validation_f0.5_score': -1, 'iterations': -1, 'error': str(e)})

    run_end_time = time.time()
    print(f"Run {i+1} time: {run_end_time - run_start_time:.2f} seconds.")


total_time = time.time() - start_time
print(f"\n--- Local Tuning Finished ---")
print(f"Total time: {total_time:.2f} seconds")

# --- 6. Post-Tuning: Threshold Adjustment and Final Evaluation ---
final_results = {}
if best_model:
    print("\n--- Selected Best Model based on Validation F0.5 Score ---")
    print(f"Best Validation F0.5 Score (@0.5 thresh): {best_objective_score:.5f}")
    print(f"Achieved at iteration: {best_iteration_count}")
    print("Best Hyperparameters:")
    print(json.dumps(best_hyperparams, indent=2))

    # --- Find Optimal Threshold on Validation Set for TARGET PRECISION ---
    optimal_threshold = find_optimal_threshold(best_model, X_val, y_val, TARGET_PRECISION)

    # --- Evaluate the BEST Model on the Held-Out TEST Set using Optimal Threshold ---
    print("\n--- Evaluating BEST Model on FINAL TEST SET using Optimal Threshold ---")
    y_pred_proba_test = best_model.predict_proba(X_test)[:, 1]
    y_pred_class_test_tuned = (y_pred_proba_test >= optimal_threshold).astype(int)

    # Log final metrics
    final_results = log_metrics(y_test, y_pred_proba_test, y_pred_class_test_tuned, prefix="final_test", threshold=optimal_threshold)
    final_results['best_hyperparameters'] = best_hyperparams
    final_results['best_objective_score_on_validation'] = best_objective_score

    # --- 7. Save the BEST Model and Threshold ---
    print(f"\n--- Saving Best Model & Threshold ---")
    try:
        joblib.dump(best_model, BEST_MODEL_SAVE_FILE)
        print(f"Best model saved to: {BEST_MODEL_SAVE_FILE}")
    except Exception as e:
        print(f"ERROR: Could not save best model: {e}")

    try:
        threshold_data = {'optimal_threshold': optimal_threshold, 'target_precision': TARGET_PRECISION}
        with open(THRESHOLD_SAVE_FILE, 'w') as f:
            json.dump(threshold_data, f, indent=4)
        print(f"Optimal threshold saved to: {THRESHOLD_SAVE_FILE}")
    except Exception as e:
        print(f"ERROR: Could not save threshold: {e}")

    try:
        with open(FINAL_EVALUATION_FILE, 'w') as f:
             # Convert numpy types for JSON serialization if necessary
            serializable_results = json.loads(json.dumps(final_results, default=lambda x: int(x) if isinstance(x, (np.int64, np.int32)) else None))
            json.dump(serializable_results, f, indent=4)
        print(f"Final evaluation metrics saved to: {FINAL_EVALUATION_FILE}")
    except Exception as e:
        print(f"ERROR: Could not save final evaluation metrics: {e}")

else:
    print("ERROR: No successful training runs completed. Could not determine best model.")

print("\n--- Local Simulation Finished ---")