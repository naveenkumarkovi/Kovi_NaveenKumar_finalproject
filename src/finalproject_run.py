"""
CS634 Final Project - Random Forest + SVM + LSTM
"""

import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
warnings.filterwarnings("ignore", category=UserWarning, module="keras")

import numpy as np
import pandas as pd

from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score, brier_score_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input
from tensorflow.keras.optimizers.legacy import Adam



# == METRICS FUNCTION ==

def compute_metrics(cm, y_true, y_prob):
    
    (TP, FN), (FP, TN) = cm
    P = TP + FN
    N = TN + FP

    def safe_div(num, den):
        return num / den if den != 0 else 0.0

    # Basic rates
    TPR = safe_div(TP, P)          
    TNR = safe_div(TN, N)          
    FPR = safe_div(FP, N)
    FNR = safe_div(FN, P)

    Precision = safe_div(TP, TP + FP)
    F1 = safe_div(2 * TP, 2 * TP + FP + FN)
    Accuracy = safe_div(TP + TN, P + N)
    Error_rate = 1.0 - Accuracy

    BACC = (TPR + TNR) / 2.0

    TSS = TPR - FPR

    HSS_num = 2 * (TP * TN - FP * FN)
    HSS_den = (P * (FN + TN) + (TP + FP) * N)
    HSS = safe_div(HSS_num, HSS_den)

    bs = brier_score_loss(y_true, y_prob)

    y_bar = np.mean(y_true)
    bs_ref = np.mean((y_true - y_bar) ** 2)
    BSS = 1.0 - safe_div(bs, bs_ref) if bs_ref != 0 else 0.0

    try:
        AUC = roc_auc_score(y_true, y_prob)
    except ValueError:
        AUC = 0.0

    return {
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "P": P, "N": N,
        "TPR": TPR, "TNR": TNR, "FPR": FPR, "FNR": FNR,
        "Precision": Precision, "Recall": TPR, "F1": F1,
        "Accuracy": Accuracy, "Error_rate": Error_rate,
        "BACC": BACC, "TSS": TSS, "HSS": HSS,
        "BS": bs, "BSS": BSS, "AUC": AUC,
    }

# == PREPROCESSING ==

def preprocess_data(df: pd.DataFrame, label_col: str):
    
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset.")

    y_raw = df[label_col]

    if not is_numeric_dtype(y_raw):
        uniques = list(y_raw.unique())
        if len(uniques) != 2:
            raise ValueError(
                f"Label column '{label_col}' has {len(uniques)} classes; "
                "script expects binary classification."
            )
        print(f"\nMapping non-numeric labels to 0/1:")
        print(f"  {uniques[0]} -> 0 (negative)")
        print(f"  {uniques[1]} -> 1 (positive)")
        mapping = {uniques[0]: 0, uniques[1]: 1}
        y = y_raw.map(mapping).astype(int).values
    else:
        uniq = sorted(y_raw.unique())
        if len(uniq) != 2:
            raise ValueError(
                f"Label column '{label_col}' has {len(uniq)} classes; "
                "script expects binary classification."
            )
        y = y_raw.astype(int).values

    X = df.drop(columns=[label_col])
    X = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).copy()

    X = X.fillna(X.median())

    X_std = (X - X.mean()) / X.std(ddof=0)
    X_std = X_std.fillna(0.0)

    feature_names = list(X_std.columns)
    return X_std.values, y, feature_names

# == LSTM MODEL BUILDER ==

def build_lstm(input_dim: int):
    """
    Simple LSTM model for tabular data.
    We treat each feature as a time step with 1 feature per step.
    """
    model = Sequential()
    model.add(Input(shape=(input_dim, 1)))
    model.add(LSTM(32))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=0.001),
        metrics=["accuracy"]
    )
    return model

# == MAIN FLOW ==

def main():

    dataset_path = input("Enter path to dataset CSV (e.g., data/diabetes.csv): ").strip()
    while not os.path.exists(dataset_path):
        print(f"\n File '{dataset_path}' not found, try again.\n")
        dataset_path = input("Enter path to dataset CSV (e.g., data/diabetes.csv): ").strip()

    print(f"\n Using dataset: {dataset_path}\n")

    df = pd.read_csv(dataset_path)
    print("Columns in dataset:\n", list(df.columns), "\n")

    label_col = input("Enter the target/label column (e.g., Outcome): ").strip()
    while label_col not in df.columns:
        print(f"\n Column '{label_col}' not found.")
        label_col = input("Enter the target/label column again: ").strip()

    folds_raw = input("Enter number of folds for Stratified K-Fold (default 10): ").strip()
    if folds_raw == "":
        n_splits = 10
    else:
        try:
            n_splits = int(folds_raw)
        except ValueError:
            print("Invalid input, using default 10.")
            n_splits = 10

    print("\nPreprocessing data...")
    X, y, feature_names = preprocess_data(df, label_col)
    print(f" Data preprocessed. {X.shape[0]} samples, {X.shape[1]} features.\n")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    rf_metrics_list = []
    svm_metrics_list = []
    lstm_metrics_list = []

    fold_num = 0
    input_dim = X.shape[1]

    for train_idx, test_idx in skf.split(X, y):
        fold_num += 1
        print(f"- Fold {fold_num}/{n_splits} - ")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # -- Random Forest --
        rf_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        rf_model.fit(X_train, y_train)

        y_pred_rf = rf_model.predict(X_test)
        y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

        cm_rf = confusion_matrix(y_test, y_pred_rf, labels=[1, 0])
        metrics_rf = compute_metrics(cm_rf, y_test, y_prob_rf)
        rf_metrics_list.append(metrics_rf)

        print("  RF   Accuracy this fold: {:.3f}".format(metrics_rf["Accuracy"]))
        print("  RF   F1       this fold: {:.3f}".format(metrics_rf["F1"]))

        # -- SVM --
        svm_model = SVC(
            kernel="linear",
            probability=True,
            random_state=42
        )
        svm_model.fit(X_train, y_train)

        y_pred_svm = svm_model.predict(X_test)
        y_prob_svm = svm_model.predict_proba(X_test)[:, 1]

        cm_svm = confusion_matrix(y_test, y_pred_svm, labels=[1, 0])
        metrics_svm = compute_metrics(cm_svm, y_test, y_prob_svm)
        svm_metrics_list.append(metrics_svm)

        print("  SVM  Accuracy this fold: {:.3f}".format(metrics_svm["Accuracy"]))
        print("  SVM  F1       this fold: {:.3f}".format(metrics_svm["F1"]))

        # -- LSTM --
        X_train_seq = X_train.reshape((X_train.shape[0], input_dim, 1))
        X_test_seq = X_test.reshape((X_test.shape[0], input_dim, 1))

        lstm_model = build_lstm(input_dim)
        lstm_model.fit(
            X_train_seq,
            y_train,
            epochs=15,
            batch_size=32,
            verbose=0
        )

        y_prob_lstm = lstm_model.predict(X_test_seq, verbose=0).ravel()
        y_pred_lstm = (y_prob_lstm >= 0.5).astype(int)

        cm_lstm = confusion_matrix(y_test, y_pred_lstm, labels=[1, 0])
        metrics_lstm = compute_metrics(cm_lstm, y_test, y_prob_lstm)
        lstm_metrics_list.append(metrics_lstm)

        print("  LSTM Accuracy this fold: {:.3f}".format(metrics_lstm["Accuracy"]))
        print("  LSTM F1       this fold: {:.3f}".format(metrics_lstm["F1"]))

    rf_df = pd.DataFrame(rf_metrics_list)
    rf_df.loc["average"] = rf_df.mean(numeric_only=True)

    svm_df = pd.DataFrame(svm_metrics_list)
    svm_df.loc["average"] = svm_df.mean(numeric_only=True)

    lstm_df = pd.DataFrame(lstm_metrics_list)
    lstm_df.loc["average"] = lstm_df.mean(numeric_only=True)

    os.makedirs("report", exist_ok=True)

    rf_path = os.path.join("report", "metrics_random_forest.csv")
    svm_path = os.path.join("report", "metrics_svm.csv")
    lstm_path = os.path.join("report", "metrics_lstm.csv")

    rf_df.to_csv(rf_path, index=True)
    svm_df.to_csv(svm_path, index=True)
    lstm_df.to_csv(lstm_path, index=True)

    print("\n Random Forest, SVM and LSTM evaluation finished.")
    print(f"RF   metrics saved to: {rf_path}")
    print(f"SVM  metrics saved to: {svm_path}")
    print(f"LSTM metrics saved to: {lstm_path}\n")

if __name__ == "__main__":
    main()
