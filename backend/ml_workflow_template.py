# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# ==========================================
# 2. DATA LOADING & CONFIGURATION
# ==========================================
# TODO: Replace 'dataset.csv' with your actual data file
# df = pd.read_csv('dataset.csv')

# For the sake of this workflow, let's assume `df` is already loaded.
# Defining feature groups based on data types
target_col = 'target'  # TODO: Replace with your actual target column name

# Identify numeric and categorical columns automatically (excluding target)
# In a real scenario, you can manually define these lists if specific handling is needed
numeric_features = ['age', 'tenure', 'balance', 'num_products'] # Example columns
categorical_features = ['geography', 'gender', 'card_type']       # Example columns

# Separate features (X) and target (y)
# X = df.drop(columns=[target_col])
# y = df[target_col]

# ==========================================
# 3. TRAIN-TEST SPLIT
# ==========================================
# Stratify ensures the class balance is maintained in both train and test sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )
# print(f"Training data shape: {X_train.shape}")
# print(f"Testing data shape: {X_test.shape}")

# ==========================================
# 4. PREPROCESSING PIPELINE
# ==========================================
# Numeric feature preprocessing: Impute missing values with median, then scale
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical feature preprocessing: Impute missing values with most frequent, then one-hot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine transformers into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop' # Drops any columns not specified in transformers
)

# ==========================================
# 5. BUILD THE FULL MACHINE LEARNING PIPELINE
# ==========================================
# Combine preprocessing and the model into a single pipeline
# This prevents data leakage during cross-validation
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

# ==========================================
# 6. CROSS VALIDATION & HYPERPARAMETER TUNING
# ==========================================
# Define the hyperparameter grid for RandomizedSearchCV
param_distributions = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

print("Workflow Template Loaded. Please provide data to begin modeling.")

# The rest of the workflow will execute once valid data is provided:
'''
print("Starting Hyperparameter Tuning...")
# RandomizedSearchCV is faster than GridSearchCV and usually finds near-optimal parameters
random_search = RandomizedSearchCV(
    model_pipeline,
    param_distributions=param_distributions,
    n_iter=10,             # Number of parameter settings that are sampled
    cv=5,                  # 5-fold cross-validation
    scoring='roc_auc',     # Optimize for Area Under ROC Curve
    verbose=1,
    random_state=42,
    n_jobs=-1              # Use all available CPU cores
)

# Train the model (this executes preprocessing + modeling + cross-validation)
random_search.fit(X_train, y_train)

print("\nBest Parameters found:")
print(random_search.best_params_)

# Extract the best model from the search
best_model = random_search.best_estimator_

# ==========================================
# 7. MODEL EVALUATION
# ==========================================
# Generate predictions on the unseen test set
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1] # Probabilities for the positive class

print("\n--- Model Performance Metrics ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# ==========================================
# 8. FEATURE IMPORTANCE (Explainability)
# ==========================================
# Extract feature names after OneHotEncoding
cat_encoder = best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
encoded_cat_features = cat_encoder.get_feature_names_out(categorical_features)
all_feature_names = numeric_features + list(encoded_cat_features)

# Extract feature importances from the Random Forest model
importances = best_model.named_steps['classifier'].feature_importances_

# Create a DataFrame for visualization
feature_importance_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\n--- Top 10 Feature Importances ---")
print(feature_importance_df.head(10))

# Plot top 10 features
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10), palette='viridis')
plt.title('Top 10 Most Important Features')
plt.tight_layout()
plt.show()

# ==========================================
# 9. SAVE THE PRODUCTION-READY MODEL
# ==========================================
model_filename = 'final_ml_pipeline.pkl'
joblib.dump(best_model, model_filename)
print(f"\\nModel pipeline successfully saved to '{model_filename}'")
# To load the model later: loaded_model = joblib.load('final_ml_pipeline.pkl')
'''
