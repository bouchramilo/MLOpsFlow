# schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional


class DiabetesInput(BaseModel):

    Pregnancies: int = Field(
        ..., 
        ge=0, 
        le=50
    )
    Glucose: float = Field(
        ..., 
        ge=0, 
        le=500
    )
    BloodPressure: float = Field(
        ..., 
        ge=0, 
        le=200,
        description="Diastolic blood pressure (mm Hg)"
    )
    SkinThickness: float = Field(
        ..., 
        ge=0, 
        le=100,
        description="Triceps skin fold thickness (mm)"
    )
    Insulin: float = Field(
        ..., 
        ge=0, 
        le=900,
        description="2-Hour serum insulin (mu U/ml)"
    )
    BMI: float = Field(
        ..., 
        ge=0, 
        le=70,
        description="Body mass index (weight in kg/(height in m)^2)"
    )
    DiabetesPedigreeFunction: float = Field(
        ..., 
        ge=0, 
        le=3,
        description="Diabetes pedigree function (genetic influence)"
    )
    Age: int = Field(
        ..., 
        ge=1, 
        le=120,
        description="Age in years"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "Pregnancies": 0,
                "Glucose": 0,
                "BloodPressure": 0,
                "SkinThickness": 0,
                "Insulin": 0,
                "BMI": 0,
                "DiabetesPedigreeFunction": 0,
                "Age": 0
            }
        }


class DiabetesBatchInput(BaseModel):
    instances: List[DiabetesInput] = Field(
        ..., 
        min_length=1,
        description="List of diabetes input instances for batch prediction"
    )


class PredictionResponse(BaseModel):
    prediction: int = Field(
        ..., 
        description="Prediction result: 0 = No Diabetes, 1 = Diabetes"
    )
    probability: Optional[List[float]] = Field(
        None, 
        description="Probability scores for each class [No Diabetes, Diabetes]"
    )
    status: str = Field(
        default="success", 
        description="Status of the prediction"
    )


class BatchPredictionResponse(BaseModel):
    predictions: List[int] = Field(
        ..., 
        description="List of prediction results"
    )
    probabilities: Optional[List[List[float]]] = Field(
        None, 
        description="List of probability scores for each instance"
    )
    count: int = Field(
        ..., 
        description="Number of predictions made"
    )
    status: str = Field(
        default="success", 
        description="Status of the batch prediction"
    )


class HealthResponse(BaseModel):
    status: str = Field(..., description="API health status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    model_version: Optional[str] = Field(None, description="Version of the loaded model")