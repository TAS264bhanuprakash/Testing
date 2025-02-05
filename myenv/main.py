from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mutual_info_score
from sklearn.ensemble import RandomForestRegressor

import io

app = FastAPI()

data = None

def json_friendly(obj):
    """Convert NumPy types to native Python types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, (np.ndarray, list, tuple, set)):
        return [json_friendly(v) for v in obj]
    return obj

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """Upload CSV file and store it globally"""
    global data
    try:
        contents = await file.read()
        data = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        data = data.replace({np.nan: None})
        return {"message": "File uploaded successfully", "columns": list(data.columns)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/granularity")
def analyze_granularity():
    """Analyze dataset granularity (rows, columns, date detection)"""
    if data is None:
        raise HTTPException(status_code=400, detail="No data uploaded")
    date_columns = [col for col in data.columns if pd.to_datetime(data[col], errors="coerce").notna().all()]
    return {
        "rows": json_friendly(len(data)),
        "columns": json_friendly(len(data.columns)),
        "date_columns": json_friendly(date_columns),
    }

@app.get("/fact-dimension")
def detect_fact_dimension():
    """Detect fact and dimension table relationships"""
    if data is None:
        raise HTTPException(status_code=400, detail="No data uploaded")
    categorical_columns = data.select_dtypes(include=["object"]).columns.tolist()
    numerical_columns = data.select_dtypes(include=["number"]).columns.tolist()
    return {
        "fact_table_columns": json_friendly(numerical_columns),
        "dimension_table_columns": json_friendly(categorical_columns),
    }

@app.get("/column-correlation")
def column_correlation():
    """Compute column correlations using a Random Forest model"""
    if data is None:
        raise HTTPException(status_code=400, detail="No data uploaded")

    df = data.copy()

    # Convert categorical columns to numerical using Label Encoding
    encoders = {}
    for col in df.select_dtypes(include=["object"]).columns:
        encoders[col] = LabelEncoder()
        df[col] = encoders[col].fit_transform(df[col].astype(str))

    # Fill missing values with median (for numerical stability)
    df.fillna(df.median(numeric_only=True), inplace=True)

    correlation_results = {}

    # Train a Random Forest model for each column
    for target_col in df.columns:
        X = df.drop(columns=[target_col])  # Features (all columns except target)
        y = df[target_col]  # Target column

        if y.nunique() == 1:  # Skip constant columns
            continue

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Get feature importance scores
        feature_importance = model.feature_importances_

        correlation_results[target_col] = {
            feature: json_friendly(importance)
            for feature, importance in zip(X.columns, feature_importance)
        }

    return correlation_results

@app.get("/data-quality")
def data_quality():
    """Compute data quality metrics (missing values, duplicates)"""
    if data is None:
        raise HTTPException(status_code=400, detail="No data uploaded")
    missing_values = data.isnull().sum().to_dict()
    duplicate_rows = data.duplicated().sum()
    return {
        "missing_values": {col: json_friendly(value) for col, value in missing_values.items()},
        "duplicate_rows": json_friendly(duplicate_rows),
    }

@app.get("/primary-foreign-keys")
def identify_keys():
    """Identify primary and potential foreign key relationships"""
    if data is None:
        raise HTTPException(status_code=400, detail="No data uploaded")
    possible_primary_keys = [col for col in data.columns if data[col].nunique() == len(data)]
    potential_foreign_keys = [col for col in data.columns if data[col].nunique() < len(data) and data[col].nunique() > 1]
    return {
        "primary_keys": json_friendly(possible_primary_keys),
        "foreign_keys": json_friendly(potential_foreign_keys),
    }

@app.get("/business-rules")
def business_rule_violations():
    """Detect violations of basic business rules"""
    if data is None:
        raise HTTPException(status_code=400, detail="No data uploaded")
    violations = {}
    if "age" in data.columns:
        violations["negative_ages"] = int((data["age"] < 0).sum())
    return {key: json_friendly(value) for key, value in violations.items()}
