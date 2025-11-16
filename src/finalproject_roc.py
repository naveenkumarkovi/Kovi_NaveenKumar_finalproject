import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
warnings.filterwarnings("ignore", category=UserWarning, module="keras")

import numpy as np
import pandas as pd

from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input
from tensorflow.keras.optimizers.legacy import Adam


import matplotlib.pyplot as plt

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

# == LSTM MODEL ==

def build_lstm(input_dim: int):
    
    model = Sequential()
    # explicit input layer (avoids Keras input_shape warning)
    model.add(Input(shape=(input_dim, 1)))
    model.add(LSTM(32))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=0.001),
        metrics=["accuracy"]
    )
    return model

# == MAIN ==

def main():

    dataset_path = input("Enter path to dataset CSV (e.g., data/diabetes.csv): ").strip()
    while not os.path.exists(dataset_path):
        print(f"\n File '{dataset_path}' not found. Try again.\n")
        dataset_path = input("Enter path to dataset CSV (e.g., data/diabetes.csv): ").strip()

    print(f"\n Using dataset: {dataset_path}\n")

    df = pd.read_csv(dataset_path)
    print("Columns in dataset:\n", list(df.columns), "\n")

    label_col = input("Enter the target/label column (e.g., Outcome): ").strip()
    while label_col not in df.columns:
        print(f"\n Column '{label_col}' not found.")
        label_col = input("Enter the target/label column again: ").strip()

    print("\nPreprocessing data...")
    X, y, feature_names = preprocess_data(df, label_col)
    print(f" Data preprocessed. {X.shape[0]} samples, {X.shape[1]} features.\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    input_dim = X_train.shape[1]

    # -- Random Forest --
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    auc_rf = roc_auc_score(y_test, y_prob_rf)

    # -- SVM --
    svm_model = SVC(
        kernel="linear",
        probability=True,
        random_state=42
    )
    svm_model.fit(X_train, y_train)
    y_prob_svm = svm_model.predict_proba(X_test)[:, 1]
    fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)
    auc_svm = roc_auc_score(y_test, y_prob_svm)

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
    fpr_lstm, tpr_lstm, _ = roc_curve(y_test, y_prob_lstm)
    auc_lstm = roc_auc_score(y_test, y_prob_lstm)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.3f})")
    plt.plot(fpr_svm, tpr_svm, label=f"SVM (AUC = {auc_svm:.3f})")
    plt.plot(fpr_lstm, tpr_lstm, label=f"LSTM (AUC = {auc_lstm:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.5)")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Diabetes Dataset")
    plt.legend(loc="lower right")
    plt.grid(True)

    os.makedirs("report", exist_ok=True)
    out_path = os.path.join("report", "roc_curves.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\n ROC curves saved to: {out_path}\n")

    plt.show()

if __name__ == "__main__":
    main()
