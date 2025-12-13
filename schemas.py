from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Union

class DatasetHealth(BaseModel):
    total_rows: int
    total_columns: int
    missing_values_count: int
    duplicate_rows: int
    health_score: float  # 0 to 100

class ColumnStats(BaseModel):
    name: str
    dtype: str
    missing_count: int
    unique_count: int
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    q1: Optional[float] = None
    median: Optional[float] = None
    q3: Optional[float] = None
    max: Optional[float] = None

class AnalysisResponse(BaseModel):
    filename: str
    health: DatasetHealth
    columns: List[ColumnStats]
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None
    head: List[Dict[str, Any]]  # First 5 rows
    sample_data: List[Dict[str, Any]] # Sample for visualizations (e.g. first 500 rows)

class AlgorithmRecommendation(BaseModel):
    algorithm: str
    reasoning: str
    type: str # 'Classification', 'Regression', 'Clustering'

class MLInsightsResponse(BaseModel):
    problem_type: str
    target_variable: Optional[str]
    recommendations: List[AlgorithmRecommendation]
    workflow: List[str] # Simple list of steps for now
