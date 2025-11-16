# CS634 – Data Mining Final Project
## Comparative Analysis of Random Forest, SVM, & LSTM for Diabetes Prediction

**Author:** Naveen Kumar Kovi - nk758  
**Course:** CS634 – Data Mining  
**Instructor:** Dr. Yasser Abdullah


## Project Overview

This project compares the performance of three **binary classification models** on the **Pima Diabetes Dataset**:

- **Random Forest (traditional ML – required)**
- **Support Vector Machine (Linear SVM)**
- **LSTM (deep learning model)**

The goal is to determine which model performs best for **diabetes prediction** using:

- 10-fold stratified cross-validation  
- Manual performance metric computation  
- Detailed ROC/AUC evaluation

## Repository Structure

```
FINALPROJECT/
│
├── README.md
│
├── data/
│   └── diabetes.csv
│
├── notebook/
│   ├── report/
│   │   ├── metrics_lstm_10fold.csv
│   │   ├── metrics_random_forest_10fold.csv
│   │   ├── metrics_svm_10fold.csv
│   └── finalproject_diabetes.ipynb
│
├── report/
│   ├── metrics_lstm.csv
│   ├── metrics_random_forest.csv
│   ├── metrics_svm.csv
│   ├── metrics_summary_auc.csv
│   └── roc_curves.png
│
└── src/
    ├── finalproject_roc.py
    └── finalproject_run.py

```

## Installation & Setup

### 1. Clone the Repository
```
git clone <your-github-repo-url>  
cd <project-folder>
```
### 2. Create the Environment
```
python3 -m venv venv  
source venv/bin/activate
```
### 3. Install Requirements
```
pip install -r requirements.txt
```

## Running the Project

### 1. Create and Activate a Virtual Environment

**Steps (for VS Code or any terminal):**

1. Open the project folder in **VS Code**.  
2. Open a **new terminal** inside VS Code (`Ctrl + ~` or *View → Terminal*).  
3. Activate your virtual environment (if not already active):
   ``` python -m venv venv ```
   ``` source venv/bin/activate  ```  

4. Install the Required Dependencies
``` pip install -r requirements.txt```

5. Run the main script:  
   ```
   python src/finalproject_run.py
   ```

**When you run this script, it will:**

- Ask you to enter the **dataset path** → e.g., `data/diabetes.csv`  
- Ask you to enter the **label column name** → e.g., `Outcome`  
- Ask for **number of folds** (press Enter for default 10)  
- Train all **three models**: Random Forest, SVM, and LSTM  
- Automatically save all generated metrics into the `report/` folder  

---

### 2. Run the ROC Curve Script (`finalproject_roc.py`)

After the training step finishes and the metrics CSV files are saved in the `report/` folder,  
you can generate ROC curves and AUC summaries.

**In the same terminal:**
```
python src/finalproject_roc.py
```

This script will:

- Read the metrics files from `/report/`  
- Compute ROC and AUC values for each model  
- Generate a combined **roc_curves.png** visualization  
- Create a summary file: `metrics_summary_auc.csv`  

---

### 3. Open the Jupyter Notebook (Optional)

If you want to explore results and visuals interactively:

jupyter notebook  

Then open:  
```notebook/finalproject_diabetes.ipynb  ```

The notebook includes:

- Exploratory Data Analysis (EDA)  
- Histograms & Correlation Heatmap  
- Class imbalance visualization  
- ROC curve plots  
- 10-fold cross-validation results  
- Per-fold and average performance tables

## Models Implemented

- Random Forest  
- Support Vector Machine (Linear)  
- LSTM Neural Network  

### Manually Computed Metrics
- TP, TN, FP, FN  
- TPR, TNR, FPR, FNR  
- Accuracy, Error Rate  
- Precision, Recall, F1  
- Balanced Accuracy  
- TSS, HSS  

### Allowed Python Packages Used For
- ROC Curve  
- AUC  
- Brier Score  
- Brier Skill Score


## Summary of Results

Based on average metrics across 10-fold cross-validation:

- **SVM →** Best overall model  
- **Random Forest →** Best recall (detecting diabetics)  
- **LSTM →** Weakest performance on this tabular dataset



## Dataset

**Dataset:** Diabetes  
**Rows:** 768 samples  
**Features:** 9 medical attributes  
**Outcome:** Binary (0 = non-diabetic, 1 = diabetic)

---

## Output Files Generated

Located inside `/report/`:

- Per-fold metrics  
- Summary metrics table  
- ROC curve image  
- Notebook-generated 10-fold metrics


## Future Improvements

- Apply SMOTE or oversampling  
- Perform feature engineering  
- Evaluate models on larger or external datasets

