# Risk Assessment in Mortgages: A Comparative Study of AI Models

This repository contains the source code and experimental notebooks used in our research to evaluate mortgage default risk using a range of AI models.

## 📊 Overview

**Objective**: Predict whether a mortgage loan will default based on structured input features like loan type, income, property value, and more.

**Model Families Tested**:
  - Artificial Neural Networks (ANN)
  - Convolutional Neural Networks (CNN)
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - K-Nearest Neighbors (KNN)
  - XGBoost

**Class Imbalance Techniques**:
- SMOTE (Synthetic Minority Oversampling Technique)
- Class weighting using scikit-learn/XGBoost options


**Evaluation Metrics**:
- Accuracy, Precision, Recall, F1-score
- Confusion Matrix
- ROC-AUC curves

## 📁 Repository Structure

```
├── EDA.ipynb                            # Exploratory Data Analysis
├── Artificial Neural Networks (ANN).ipynb
├── Convolutional Neural Network (CNN).ipynb
├── Logistic Regression.ipynb
├── Decision Tree.ipynb
├── Random Forest.ipynb
├── KNN.ipynb
├── XGBoost.ipynb
├── Loan_Default.csv                    # Raw dataset
├── Loan_Default_Cleaned.csv            # Cleaned/preprocessed dataset
├── requirements.txt                    # Python dependencies
├── README.md
```

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/bgosal/Mortgage-Risk-Assessment-.git
cd Mortgage-Risk-Assessment-
```

### 2. Create and activate a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook
```bash
jupyter notebook
```

Then open any model notebook such as:
- `Convolutional Neural Network (CNN).ipynb`
- `XGBoost.ipynb`
- `Random Forest.ipynb`

Run all cells from top to bottom to train the model and view evaluation outputs.

---
