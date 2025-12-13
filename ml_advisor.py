import pandas as pd
import numpy as np
from typing import List, Tuple
from ..schemas import AlgorithmRecommendation

def detect_problem_type(df: pd.DataFrame, target_col: str = None) -> Tuple[str, str]:
    """
    Returns (ProblemType, TargetColumnName)
    """
    # 1. If user selects target -> use it
    if target_col and target_col in df.columns:
        y = df[target_col]
        if is_classification_target(y):
            return "Classification", target_col
        return "Regression", target_col

    # 2. Look for auto-detect columns
    potential_targets = ['target', 'class', 'label', 'outcome', 'survived', 'churn']
    for col in potential_targets:
        # Case-insensitive match
        match = next((c for c in df.columns if c.lower() == col), None)
        if match:
            y = df[match]
            if is_classification_target(y):
                return "Classification", match
            # If named 'target' but is float, likely regression
            if np.issubdtype(y.dtype, np.number):
                return "Regression", match

    # 3. Else -> Clustering
    return "Clustering", None

def is_classification_target(series: pd.Series) -> bool:
    """
    Heuristic: 
    - Binary (<=2 unique) -> Classification
    - Object/Category dtype -> Classification
    - Integer with few unique values (<10) -> Classification
    """
    if series.dtype == 'object' or str(series.dtype) == 'category':
        return True
    
    unique_count = series.nunique()
    if unique_count <= 2:
        return True
    
    # If integer and small number of classes, treat as classification
    if np.issubdtype(series.dtype, np.integer) and unique_count < 15:
        return True
        
    return False

def check_imbalance(series: pd.Series) -> bool:
    """
    Check if class imbalance exists (e.g., one class < 20% of data)
    """
    if series is None: return False
    value_counts = series.value_counts(normalize=True)
    min_class_pct = value_counts.min()
    return min_class_pct < 0.20 # Less than 20% representation

def recommend_algorithms(df: pd.DataFrame, problem_type: str, target_col: str = None) -> List[AlgorithmRecommendation]:
    recs = []
    rows, cols = df.shape
    
    # Contextual hints
    is_imbalanced = False
    if target_col and target_col in df.columns:
        is_imbalanced = check_imbalance(df[target_col])
    
    # --- CLASSIFICATION ---
    if problem_type == "Classification":
        # Logistic Regression
        recs.append(AlgorithmRecommendation(
            algorithm="Logistic Regression",
            reasoning="Strong baseline for binary classification and interpretable results. Works well when relationships are linear.",
            type="Classification"
        ))
        
        # Random Forest
        rf_reason = "Handles non-linear feature interactions well."
        if is_imbalanced:
            rf_reason += " and class imbalance effectively."
        recs.append(AlgorithmRecommendation(
            algorithm="Random Forest Classifier",
            reasoning=rf_reason,
            type="Classification"
        ))
        
        # XGBoost
        xgb_reason = "Industry-standard for high performance."
        if is_imbalanced:
            xgb_reason = "Industry-standard for fraud detection/imbalance. Performs well on imbalanced datasets."
        recs.append(AlgorithmRecommendation(
            algorithm="XGBoost Classifier",
            reasoning=xgb_reason,
            type="Classification"
        ))

    # --- REGRESSION ---
    elif problem_type == "Regression":
        recs.append(AlgorithmRecommendation(
            algorithm="Linear Regression",
            reasoning="Baseline model. Best if the target has a linear relationship with features.",
            type="Regression"
        ))
        
        recs.append(AlgorithmRecommendation(
            algorithm="Random Forest Regressor",
            reasoning="Robust to outliers and captures non-linear patterns without heavy feature scaling.",
            type="Regression"
        ))
        
        if rows > 1000:
            recs.append(AlgorithmRecommendation(
                algorithm="Gradient Boosting Regressor (XGBoost)",
                reasoning="High accuracy on structured data. Minimizes loss effectively through iterative boosting.",
                type="Regression"
            ))

    # --- CLUSTERING ---
    elif problem_type == "Clustering":
        recs.append(AlgorithmRecommendation(
            algorithm="K-Means Clustering",
            reasoning="Standard algorithm for grouping data into K distinct clusters. Efficient for medium datasets.",
            type="Clustering"
        ))
        
        recs.append(AlgorithmRecommendation(
            algorithm="DBSCAN",
            reasoning="Density-based clustering. Good for finding outliers and clusters of arbitrary shape.",
            type="Clustering"
        ))
        
        
    return recs

def generate_workflow(df: pd.DataFrame, problem_type: str, target_col: str = None) -> List[str]:
    workflow = []
    
    # 1. Handling Missing Values
    missing_sum = df.isnull().sum().sum()
    if missing_sum > 0:
        workflow.append("Missing Values: Impute numerical columns with Median and categorical with Mode, or drop rows if missingness is low (<5%).")
    else:
        workflow.append("Missing Values: None detected. No imputation required.")
        
    # 2. Handling Duplicates
    if df.duplicated().sum() > 0:
        workflow.append("Duplicates: Remove duplicate rows to prevent data leakage and bias.")
    else:
        workflow.append("Duplicates: No duplicates found. Dataset is clean.")
        
    # 3. Outlier Treatment (Simple Heuristic)
    # If numeric cols exist, suggest scaler
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
         workflow.append("Outliers & Scaling: Apply RobustScaler to handle potential outliers in numeric features.")
    
    # 4. Preprocessing
    workflow.append("Preprocessing: Encode categorical variables using One-Hot Encoding (for nominal) or Label Encoding (for ordinal).")
    
    # 5. Training & Eval
    if problem_type == "Classification":
        workflow.append("Model Training: Split data 80/20. Train models using Stratified K-Fold Cross Validation.")
        workflow.append("Evaluation: Optimize for F1-Score or AUC-ROC, especially if classes are imbalanced.")
    elif problem_type == "Regression":
         workflow.append("Model Training: Split data 80/20. Use Cross Validation to ensure generalizability.")
         workflow.append("Evaluation: Use RMSE or MAE to minimize prediction error. Check R-Squared for goodness of fit.")
    else:
         workflow.append("Clustering: Use Elbow Method to determine optimal K (clusters).")
         workflow.append("Evaluation: Use Silhouette Score to measure cluster separation.")
         
    return workflow
