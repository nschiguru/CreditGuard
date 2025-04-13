import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             precision_recall_curve, auc, f1_score, precision_score, recall_score,
                             fbeta_score, make_scorer)
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import sys
import joblib
import time
import json
import os
from scipy.stats import uniform, randint

print("--- Local XGBoost Tuning with RandomizedSearchCV & Threshold Optimization ---")
print("--- Goal: Achieve >= 90% Precision on Test Set ---")

# --- Configuration ---
TRAINING_DATA_FILE = 'training_data_unscaled.csv'
BEST_MODEL_SAVE_FILE = 'creditguard_xgb_best_randomsearch.joblib'
THRESHOLD_SAVE_FILE = 'optimal_threshold_randomsearch.json'
FINAL_EVALUATION_FILE = 'final_evaluation_randomsearch.json'

TARGET_PRECISION = 0.90
RANDOM_STATE = 42

TEST_SET_SIZE = 0.20
N_ITER_SEARCH = 30
CV_FOLDS = 3

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


    valid_indices = [i for i, p in enumerate(precision_val) if p >= target_precision and i < len(thresholds_val)]

    if valid_indices:
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
    return optimal_threshold


# --- 1. Load Data ---
print(f"\nLoading data from: {TRAINING_DATA_FILE}")
try:
    data = pd.read_csv(TRAINING_DATA_FILE)
    print(f"Loaded data shape: {data.shape}")
    if data.empty: sys.exit("Error: Loaded data is empty.")
except Exception as e:
    print(f"ERROR loading data: {e}"); sys.exit(1)

target_col = data.columns[-1]
X_full = data.drop(target_col, axis=1)
y_full = data[target_col].astype(int)

print(f"\nSplitting off Test set ({TEST_SET_SIZE*100:.0f}%)...")
X_train_cv, X_test, y_train_cv, y_test = train_test_split(
    X_full, y_full, test_size=TEST_SET_SIZE, random_state=RANDOM_STATE, stratify=y_full
)
print(f"Train/CV set shape: {X_train_cv.shape} (Fraud cases: {sum(y_train_cv)})")
print(f"Test set shape:     {X_test.shape} (Fraud cases: {sum(y_test)})")

# --- 3. Calculate scale_pos_weight (on the full Train+CV set) ---
print("\nCalculating scale_pos_weight on training data...")
neg_count, pos_count = np.bincount(y_train_cv)
scale_pos_weight_value = neg_count / pos_count if pos_count > 0 else 1
print(f'Calculated scale_pos_weight: {scale_pos_weight_value:.2f}')

# --- 4. Define Model and Hyperparameter Search Space ---
# Base XGBoost model
xgb_clf = xgb.XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=scale_pos_weight_value,
    use_label_encoder=False,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    # Use early stopping during RandomizedSearchCV fit
    # eval_metric will be set internally by scorer, but good practice
    eval_metric='aucpr'
)

# Hyperparameter distribution for RandomizedSearchCV
param_dist = {
    'n_estimators': randint(100, 500),
    'learning_rate': uniform(0.01, 0.29),
    'max_depth': randint(3, 9),
    'subsample': uniform(0.6, 0.35),
    'colsample_bytree': uniform(0.6, 0.35),
    'gamma': uniform(0.0, 0.5),
    'reg_alpha': uniform(0.0, 0.5),
    'reg_lambda': uniform(0.5, 1.5)
}

f05_scorer = make_scorer(fbeta_score, beta=0.5)

cv_strategy = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

print(f"\n--- Starting Randomized Search (Iterations={N_ITER_SEARCH}, CV Folds={CV_FOLDS}) ---")
print(f"--- Optimizing for: F0.5 Score ---")
start_time = time.time()

random_search = RandomizedSearchCV(
    estimator=xgb_clf,
    param_distributions=param_dist,
    n_iter=N_ITER_SEARCH,
    scoring=f05_scorer, # Use F0.5 score for optimization
    cv=cv_strategy,
    n_jobs=-1, # Use all available cores for CV folds if possible
    verbose=2, # Show progress
    random_state=RANDOM_STATE,
    refit=True 
)


total_time = time.time() - start_time
print(f"\n--- Randomized Search Finished ---")
print(f"Total time: {total_time:.2f} seconds")

# --- 7. Get Best Model and Hyperparameters ---
print("\n--- Best Model Found by Randomized Search ---")
print(f"Best F0.5 Score (averaged across CV folds): {random_search.best_score_:.5f}")
print("Best Hyperparameters:")
best_params = random_search.best_params_
print(json.dumps(best_params, indent=2))

# The best model is already refitted on the entire X_train_cv, y_train_cv data
best_model = random_search.best_estimator_

# --- 8. Post-Tuning: Threshold Adjustment and Final Evaluation ---
final_results = {}
if best_model:
    optimal_threshold = find_optimal_threshold(best_model, X_train_cv, y_train_cv, TARGET_PRECISION)

    # --- Evaluate the BEST Model on the Held-Out TEST Set using Optimal Threshold ---
    print("\n--- Evaluating BEST Model on FINAL TEST SET using Optimal Threshold ---")
    y_pred_proba_test = best_model.predict_proba(X_test)[:, 1]
    y_pred_class_test_tuned = (y_pred_proba_test >= optimal_threshold).astype(int)

    # Log final metrics
    final_results = log_metrics(y_test, y_pred_proba_test, y_pred_class_test_tuned, prefix="final_test", threshold=optimal_threshold)
    final_results['best_hyperparameters'] = best_params
    final_results['best_cv_score_f0.5'] = random_search.best_score_

    # --- 9. Save the BEST Model and Threshold ---
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
            serializable_results = json.loads(json.dumps(final_results, default=lambda x: int(x) if isinstance(x, (np.int64, np.int32)) else None))
            json.dump(serializable_results, f, indent=4)
        print(f"Final evaluation metrics saved to: {FINAL_EVALUATION_FILE}")
    except Exception as e:
        print(f"ERROR: Could not save final evaluation metrics: {e}")

else:
    print("ERROR: RandomizedSearch did not yield a best model.")

print("\n--- Local Simulation with RandomizedSearch Finished ---")
