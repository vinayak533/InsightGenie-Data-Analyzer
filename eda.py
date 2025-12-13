import pandas as pd
import numpy as np
from typing import List, Dict, Any
from ..schemas import ColumnStats, DatasetHealth

def calculate_health(df: pd.DataFrame) -> DatasetHealth:
    total_rows = len(df)
    total_cols = len(df.columns)
    missing_count = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()
    
    # Realistic Health Score Logic
    # Base Score: 100
    # Formula requested: No Nulls (+40%), Valid Schema (+30%, assumed true if read), Duplicates (-5 to -10%)
    # Let's pivot to a deduction model from 100 to match the "Max 100" constraint, 
    # OR implement exactly as requested: 
    # Start 0? No, usually health implies start at 100 or start at 0 and add up. 
    # "No nulls -> +40%" implies components.
    
    score = 0
    
    # 1. Schema/Structure Validity (30%) - If we opened it, it's valid.
    score += 30
    
    # 2. Completeness (40%)
    if missing_count == 0:
        score += 40
    else:
        # Partial credit: (1 - missing_ratio) * 40
        missing_ratio = missing_count / (total_rows * total_cols)
        score += max(0, (1 - missing_ratio * 5) * 40) # penalize heavily for any missing
        
    # 3. Uniqueness (30%)
    # Deduct for duplicates
    if duplicate_rows > 0:
        # User said "Duplicates present -> -5 to -10%"
        # Let's say if we have < 5% duplicates, -5. If > 5%, -10.
        duplicate_ratio = duplicate_rows / total_rows
        deduction = 10 if duplicate_ratio > 0.05 else 5
        score += (30 - deduction)
    else:
        score += 30

    return DatasetHealth(
        total_rows=total_rows,
        total_columns=total_cols,
        missing_values_count=int(missing_count),
        duplicate_rows=int(duplicate_rows),
        health_score=min(100.0, round(score, 1))
    )

def generate_summary_stats(df: pd.DataFrame) -> List[ColumnStats]:
    stats = []
    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)
        is_numeric = np.issubdtype(series.dtype, np.number)
        
        col_stat = ColumnStats(
            name=col,
            dtype=dtype,
            missing_count=int(series.isnull().sum()),
            unique_count=int(series.nunique())
        )
        
        if is_numeric:
            col_stat.mean = float(series.mean()) if not series.isnull().all() else None
            col_stat.std = float(series.std()) if not series.isnull().all() else None
            col_stat.min = float(series.min()) if not series.isnull().all() else None
            col_stat.max = float(series.max()) if not series.isnull().all() else None
            col_stat.median = float(series.median()) if not series.isnull().all() else None
            col_stat.q1 = float(series.quantile(0.25)) if not series.isnull().all() else None
            col_stat.q3 = float(series.quantile(0.75)) if not series.isnull().all() else None
            
        stats.append(col_stat)
    return stats

def get_correlation_matrix(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    # Only numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return {}
    
    corr = numeric_df.corr().fillna(0)
    # Convert to dict of dicts for JSON serialization
    return corr.to_dict()

def get_head(df: pd.DataFrame, n: int = 5) -> List[Dict[str, Any]]:
    # Replace NaN with None for JSON compatibility
    return df.head(n).replace({np.nan: None}).to_dict(orient='records')

def get_sample(df: pd.DataFrame, n: int = 500) -> List[Dict[str, Any]]:
    # Return a sample for plotting
    limit = min(len(df), n)
    # Simple top-N for now to avoid shuffling overhead on large files, or sample?
    # Sampling is better for distribution.
    if len(df) > n:
        return df.sample(n=n).replace({np.nan: None}).to_dict(orient='records')
    return df.head(n).replace({np.nan: None}).to_dict(orient='records')
