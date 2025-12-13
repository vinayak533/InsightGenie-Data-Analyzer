from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
from .schemas import AnalysisResponse, MLInsightsResponse
from .services.eda import generate_summary_stats, calculate_health, get_correlation_matrix, get_head, get_sample
from .services.ml_advisor import detect_problem_type, recommend_algorithms, generate_workflow
from pydantic import ValidationError

app = FastAPI(title="InsightGenie API", description="Backend for InsightGenie Analytics Platform", version="1.0.0")

# CORS Setup
origins = ["http://localhost:5173", "http://localhost:3000", "*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "InsightGenie API is running"}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Basic EDA
        health = calculate_health(df)
        stats = generate_summary_stats(df)
        correlation = get_correlation_matrix(df)
        head_data = get_head(df)
        sample_data = get_sample(df) # Get sample for charts
        
        return AnalysisResponse(
            filename=file.filename,
            health=health,
            columns=stats,
            correlation_matrix=correlation,
            head=head_data,
            sample_data=sample_data
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/recommend", response_model=MLInsightsResponse)
async def get_recommendations(
    target_column: str = Form(None),
    file: UploadFile = File(...)
):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # ML Advice
        problem_type, detected_target = detect_problem_type(df, target_column)
        recommendations = recommend_algorithms(df, problem_type, detected_target)
        workflow = generate_workflow(df, problem_type, detected_target)
        
        # Update target_column in response if it was auto-detected
        final_target = detected_target if detected_target else target_column
        
        return MLInsightsResponse(
            problem_type=problem_type,
            target_variable=final_target,
            recommendations=recommendations,
            workflow=workflow
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")
