# Diabetes Prediction Using Machine Learning

A complete machine learning project for predicting diabetes using the Pima Indians Diabetes Dataset. The project covers data loading, exploratory data analysis, preprocessing, model training, hyperparameter tuning, neural network training, model evaluation, result visualization, and SHAP-based explainability.

## Project Overview

This project predicts whether a patient is likely to have diabetes based on medical diagnostic measurements such as glucose level, BMI, blood pressure, insulin, age, and other health indicators.

The target column is `Outcome`:

- `0` means the patient is not diabetic
- `1` means the patient is diabetic

The main implementation is in:

```text
diabetes_prediction_system.py
```

## Dataset

The project uses the Pima Indians Diabetes Dataset, stored locally at:

```text
Dataset/diabetes.csv
```

Dataset size:

- 768 samples
- 8 input features
- 1 target column

Features used:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

Target:

- Outcome

## Key Features

- Data loading and inspection
- Missing/invalid value handling
- Exploratory data analysis
- Feature scaling using `StandardScaler`
- Train/test split with stratification
- Machine learning model comparison
- Hyperparameter tuning with `GridSearchCV`
- Artificial Neural Network implementation
- Confusion matrix generation
- ROC curve comparison
- Model performance reports
- SHAP explainability analysis
- Saved trained models for reuse
- Generated plots and result files

## Models Used

The project trains and compares the following models:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine
- K-Nearest Neighbors
- Artificial Neural Network

## Best Model

Based on the current saved results, the best performing model is:

```text
Artificial Neural Network
```

Performance:

```text
Accuracy: 0.7403
ROC-AUC: 0.8204
```

## Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.7013 | 0.5870 | 0.5000 | 0.5400 | 0.8128 |
| Decision Tree | 0.6818 | 0.5532 | 0.4815 | 0.5149 | 0.6357 |
| Random Forest | 0.7792 | 0.7273 | 0.5926 | 0.6531 | 0.8191 |
| SVM | 0.7078 | 0.6047 | 0.4815 | 0.5361 | 0.8130 |
| KNN | 0.7338 | 0.6327 | 0.5741 | 0.6019 | 0.7919 |
| ANN | 0.7403 | 0.6458 | 0.5741 | 0.6078 | 0.8204 |

Full comparison is available in:

```text
results/model_comparison.csv
```

## SHAP Explainability

SHAP is used to explain model predictions and identify the most influential features.

Top influential features:

```text
Glucose, BMI, Age
```

The most important feature in the current analysis is:

```text
Glucose
```

SHAP outputs are saved in:

```text
results/shap_feature_importance.csv
plots/shap_summary_bar.png
plots/shap_summary_beeswarm.png
plots/shap_waterfall_plot.png
```

## Project Structure

```text
Diabetes/
├── Dataset/
│   └── diabetes.csv
├── Docs/
│   ├── Report.docx
│   ├── Research Article.docx
│   └── Sample Article.pdf
├── plots/
│   ├── class_distribution.png
│   ├── correlation_heatmap.png
│   ├── roc_curve_comparison.png
│   ├── confusion_matrix_*.png
│   ├── comparison_*.png
│   └── shap_*.png
├── results/
│   ├── final_summary.txt
│   ├── model_comparison.csv
│   ├── report_*.txt
│   └── shap_feature_importance.csv
├── saved_models/
│   ├── ann_model.h5
│   └── *.joblib
├── diabetes_prediction_system.py
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Azeem-Ramay/Diabetes-Prediction-Using-Machine-Learning.git
cd Diabetes-Prediction-Using-Machine-Learning
```

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install the required Python packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow shap joblib
```

## How to Run

Run the main script:

```bash
python diabetes_prediction_system.py
```

After execution, the script will:

- Train all models
- Save trained models in `saved_models/`
- Save reports in `results/`
- Save visualizations in `plots/`
- Print the final model comparison and best model summary

## Outputs

Generated result files:

```text
results/final_summary.txt
results/model_comparison.csv
results/report_logistic_regression.txt
results/report_decision_tree.txt
results/report_random_forest.txt
results/report_svm.txt
results/report_knn.txt
results/shap_feature_importance.csv
```

Generated visualizations include:

- Class distribution chart
- Correlation heatmap
- Feature histograms
- Feature boxplots
- Confusion matrices
- Model comparison bar charts
- ROC curve comparison
- ANN training history
- SHAP feature importance plots
- SHAP dependence plots
- SHAP waterfall plot

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras
- SHAP
- Joblib

## Notes

This project is intended for educational and research purposes. It demonstrates a complete machine learning workflow for a healthcare-related classification problem, but it should not be used as a substitute for professional medical diagnosis.

## Author

Developed by Azeem Ramay.
