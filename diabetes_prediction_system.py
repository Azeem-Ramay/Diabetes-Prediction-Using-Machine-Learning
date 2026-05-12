"""
Diabetes Prediction Machine Learning Project
Developed for: Pima Indians Diabetes Dataset
Features: Data Cleaning, EDA, Model Comparison, Hyperparameter Tuning, ANN, SHAP Explainability, and Visualization.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# TensorFlow / Keras imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Explainability imports
import shap

# Create directories if they don't exist
for folder in ['plots', 'results', 'saved_models']:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Set visual style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ==========================================
# Step 2 — Load Dataset
# ==========================================
print("--- Loading Dataset ---")
df = pd.read_csv('Dataset/diabetes.csv')

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nDataset Info:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

# ==========================================
# Step 3 — Exploratory Data Analysis (EDA)
# ==========================================
print("\n--- Performing EDA ---")

# 1. Class distribution
plt.figure()
sns.countplot(x='Outcome', hue='Outcome', data=df, palette='viridis', legend=False)
plt.title('Diabetes Outcome Distribution (0: No, 1: Yes)')
plt.savefig('plots/class_distribution.png')
plt.close()

# 2. Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.savefig('plots/correlation_heatmap.png')
plt.close()

# 3. Histograms
df.hist(bins=20, figsize=(20, 15))
plt.suptitle('Feature Histograms')
plt.savefig('plots/feature_histograms.png')
plt.close()

# 4. Boxplots
plt.figure(figsize=(20, 10))
sns.boxplot(data=df, orient="h", palette="Set2")
plt.title('Boxplot of Features')
plt.savefig('plots/feature_boxplots.png')
plt.close()

# ==========================================
# Step 4 — Data Preprocessing
# ==========================================
print("\n--- Data Preprocessing ---")

# Replace invalid zero values with median
cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
print(f"Replacing zeros in {cols_to_fix} with median values...")

for col in cols_to_fix:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

# Split Features and Target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# Step 5 — Train-Test Split
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Split completed: Train={len(X_train)}, Test={len(X_test)}")

# ==========================================
# Bonus: Hyperparameter Tuning
# ==========================================
print("\n--- Hyperparameter Tuning ---")

# 1. Random Forest
rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, scoring='accuracy')
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_
print(f"Best RF Params: {rf_grid.best_params_}")

# 2. SVM
svm_params = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'], 'kernel': ['linear', 'rbf']}
svm_grid = GridSearchCV(SVC(probability=True, random_state=42), svm_params, cv=5, scoring='accuracy')
svm_grid.fit(X_train, y_train)
best_svm = svm_grid.best_estimator_
print(f"Best SVM Params: {svm_grid.best_params_}")

# 3. KNN
knn_params = {'n_neighbors': [3, 5, 7, 9, 11], 'weights': ['uniform', 'distance']}
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5, scoring='accuracy')
knn_grid.fit(X_train, y_train)
best_knn = knn_grid.best_estimator_
print(f"Best KNN Params: {knn_grid.best_params_}")

# ==========================================
# Step 6 — Model Training & CV
# ==========================================
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': best_rf,
    'SVM': best_svm,
    'KNN': best_knn
}

results_list = []

print("\n--- Training and Evaluating Sklearn Models ---")

for name, model in models.items():
    # Cross Validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5)
    avg_cv = cv_scores.mean()
    
    # Fit on training set
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    results_list.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': auc,
        'CV_Avg_Accuracy': avg_cv
    })
    
    print(f"{name} -> Accuracy: {acc:.4f}, AUC: {auc:.4f}, CV: {avg_cv:.4f}")
    
    # Confusion Matrix Plot
    plt.figure()
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'plots/confusion_matrix_{name.lower().replace(" ", "_")}.png')
    plt.close()
    
    # Save sklearn models
    joblib.dump(model, f'saved_models/{name.lower().replace(" ", "_")}.joblib')
    
    # Save classification report
    with open(f'results/report_{name.lower().replace(" ", "_")}.txt', 'w') as f:
        f.write(f"Classification Report for {name}:\n")
        f.write(classification_report(y_test, y_pred))

# ==========================================
# Step 6 (Cont.) — ANN Implementation
# ==========================================
print("\n--- Training ANN ---")

ann_model = Sequential([
    Dense(16, input_dim=X_train.shape[1], activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = ann_model.fit(
    X_train, y_train, 
    validation_data=(X_test, y_test),
    epochs=50, 
    batch_size=16, 
    verbose=0
)

# Evaluate ANN
y_prob_ann = ann_model.predict(X_test).flatten()
y_pred_ann = (y_prob_ann > 0.5).astype(int)

acc_ann = accuracy_score(y_test, y_pred_ann)
prec_ann = precision_score(y_test, y_pred_ann)
rec_ann = recall_score(y_test, y_pred_ann)
f1_ann = f1_score(y_test, y_pred_ann)
auc_ann = roc_auc_score(y_test, y_prob_ann)

results_list.append({
    'Model': 'ANN',
    'Accuracy': acc_ann,
    'Precision': prec_ann,
    'Recall': rec_ann,
    'F1-Score': f1_ann,
    'ROC-AUC': auc_ann,
    'CV_Avg_Accuracy': np.nan
})

print(f"ANN -> Accuracy: {acc_ann:.4f}, AUC: {auc_ann:.4f}")

# ANN Accuracy/Loss Graphs
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('ANN Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('ANN Loss')
plt.legend()
plt.savefig('plots/ann_training_history.png')
plt.close()

# ANN Confusion Matrix
plt.figure()
cm_ann = confusion_matrix(y_test, y_pred_ann)
sns.heatmap(cm_ann, annot=True, fmt='d', cmap='Reds')
plt.title('Confusion Matrix: ANN')
plt.savefig('plots/confusion_matrix_ann.png')
plt.close()

# Save ANN model
ann_model.save('saved_models/ann_model.h5')

# ==========================================
# Step 8 — ROC Curve Comparison
# ==========================================
plt.figure(figsize=(10, 8))

for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_val:.3f})')

# Add ANN to ROC
fpr_ann, tpr_ann, _ = roc_curve(y_test, y_prob_ann)
plt.plot(fpr_ann, tpr_ann, label=f'ANN (AUC = {auc_ann:.3f})', linestyle='--')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.savefig('plots/roc_curve_comparison.png')
plt.close()

# ==========================================
# Step 10 & 11 — Comparison and Graphs
# ==========================================
results_df = pd.DataFrame(results_list)
results_df.to_csv('results/model_comparison.csv', index=False)

# Bar chart comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
for metric in metrics:
    plt.figure()
    sns.barplot(x='Model', y=metric, hue='Model', data=results_df, palette='magma', legend=False)
    plt.title(f'Model Comparison: {metric}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'plots/comparison_{metric.lower()}.png')
    plt.close()

# ==========================================
# Step 12 — Best Model Detection
# ==========================================
best_row = results_df.loc[results_df['ROC-AUC'].idxmax()]
print("\n" + "="*30)
print(f"BEST MODEL DETECTED: {best_row['Model']}")
print(f"Best ROC-AUC Score: {best_row['ROC-AUC']:.4f}")
print("="*30)

# ==========================================
# Additional Requirement — SHAP Explainability
# ==========================================
print("\n--- Initializing SHAP Explainability ---")

# Step 1: Select Best sklearn Model
sklearn_best_name = results_df[results_df['Model'] != 'ANN'].loc[results_df[results_df['Model'] != 'ANN']['ROC-AUC'].idxmax()]['Model']
best_model = models[sklearn_best_name]

print(f"Generating SHAP explanations for the best sklearn model: {sklearn_best_name}")

# Step 2: Initialize Explainer
tree_models = ['Random Forest', 'Decision Tree']
if sklearn_best_name in tree_models:
    explainer = shap.TreeExplainer(best_model)
else:
    explainer = shap.Explainer(best_model, X_train)

# Step 3: Generate SHAP Values
shap_values = explainer(X_test)

# Fix for "beeswarm plot does not support plotting explanations with instances that have more than one dimension"
# For binary classification, SHAP often returns values for both classes [0, 1]. We select class 1.
if len(shap_values.shape) == 3:
    print("Selecting SHAP values for the positive class (Class 1)...")
    shap_values = shap_values[:, :, 1]

# Step 4: SHAP Summary Plot (Bar)
plt.figure()
shap.summary_plot(shap_values, X_test, feature_names=X.columns, plot_type="bar", show=False)
plt.title(f'SHAP Feature Importance (Bar) - {sklearn_best_name}')
plt.tight_layout()
plt.savefig('plots/shap_summary_bar.png')
plt.close()

# SHAP Summary Plot (Beeswarm)
plt.figure()
shap.plots.beeswarm(shap_values, show=False)
plt.title(f'SHAP Summary Beeswarm Plot - {sklearn_best_name}')
plt.tight_layout()
plt.savefig('plots/shap_summary_beeswarm.png')
plt.close()

# Step 5: SHAP Feature Importance CSV
if isinstance(shap_values, shap._explanation.Explanation):
    vals = np.abs(shap_values.values).mean(0)
else:
    vals = np.abs(shap_values).mean(0)

shap_importance_df = pd.DataFrame(list(zip(X.columns, vals)), columns=['Feature Name', 'Mean Absolute SHAP Value'])
shap_importance_df.sort_values(by=['Mean Absolute SHAP Value'], ascending=False, inplace=True)
shap_importance_df.to_csv('results/shap_feature_importance.csv', index=False)

# Step 6: SHAP Dependence Plots
top_features = ['Glucose', 'BMI', 'Age']
for feat in top_features:
    plt.figure()
    feat_idx = list(X.columns).index(feat)
    # Use shap_values.values for dependence_plot
    shap.dependence_plot(feat_idx, shap_values.values, X_test, feature_names=X.columns, show=False)
    plt.title(f'SHAP Dependence Plot: {feat}')
    plt.savefig(f'plots/shap_dependence_{feat.lower()}.png')
    plt.close()

# Step 8: SHAP Waterfall Plot for a single prediction
plt.figure()
shap.plots.waterfall(shap_values[0], show=False)
plt.title(f'SHAP Waterfall Plot (Sample 0) - {sklearn_best_name}')
plt.tight_layout()
plt.savefig('plots/shap_waterfall_plot.png')
plt.close()

# Step 9: Interpretation Section
print("\n--- SHAP Interpretation Summary ---")
print(f"1. The most influential features for {sklearn_best_name} are: {', '.join(shap_importance_df['Feature Name'].head(3).values)}")
print(f"2. {shap_importance_df.iloc[0]['Feature Name']} has the strongest impact on prediction outcomes.")
print("3. SHAP values provide transparency by showing how each feature pushes the model prediction away from the baseline.")
print("4. This explainability is crucial for healthcare AI, allowing clinicians to validate model reasoning.")

# ==========================================
# Step 14 — Final Summary
# ==========================================
summary_text = f"""
--- Final Professional Summary ---
Dataset: Pima Indians Diabetes Dataset
Dataset Size: {df.shape[0]} samples, {df.shape[1]} features
Train/Test Split: 80% / 20%
Best Performing Model: {best_row['Model']}
Best ROC-AUC: {best_row['ROC-AUC']:.4f}
Best Accuracy: {best_row['Accuracy']:.4f}

SHAP Analysis:
- Most Important Feature: {shap_importance_df.iloc[0]['Feature Name']}
- Top 3 Influences: {', '.join(shap_importance_df['Feature Name'].head(3).values)}

Model comparison saved to: results/model_comparison.csv
Trained models saved to: saved_models/
All visualizations (including SHAP) saved to: plots/
Explainability report saved to: results/shap_feature_importance.csv
"""
print(summary_text)

with open('results/final_summary.txt', 'w') as f:
    f.write(summary_text)
    f.write("\n\nFull Comparison:\n")
    f.write(results_df.to_string())

print("\nProject execution complete successfully.")
