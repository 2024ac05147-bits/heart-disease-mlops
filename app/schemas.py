"""Pydantic request and response schemas for the prediction API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class HeartDiseaseFeatures(BaseModel):
    """Input features required by the trained model."""

    age: float = Field(
        ...,
        ge=18,
        le=120,
        description="Age in years.",
        examples=[57],
    )
    sex: int = Field(
        ...,
        ge=0,
        le=1,
        description="Sex category: 0=female, 1=male.",
        examples=[1],
    )
    cp: int = Field(
        ...,
        ge=1,
        le=4,
        description="Chest pain category, encoded from 1 to 4.",
        examples=[4],
    )
    trestbps: float = Field(
        ...,
        gt=0,
        le=300,
        description="Resting blood pressure in mm Hg.",
        examples=[140],
    )
    chol: float = Field(
        ...,
        gt=0,
        le=1000,
        description="Serum cholesterol in mg/dl.",
        examples=[241],
    )
    fbs: int | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Fasting blood sugar category.",
        examples=[0],
    )
    restecg: int | None = Field(
        default=None,
        ge=0,
        le=2,
        description="Resting ECG category.",
        examples=[1],
    )
    thalach: float | None = Field(
        default=None,
        ge=40,
        le=250,
        description="Maximum achieved heart rate.",
        examples=[123],
    )
    exang: int | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Exercise-induced angina category.",
        examples=[1],
    )
    oldpeak: float | None = Field(
        default=None,
        ge=-5,
        le=10,
        description="ST depression induced by exercise.",
        examples=[0.2],
    )
    slope: int | None = Field(
        default=None,
        ge=1,
        le=3,
        description="Slope of the peak exercise ST segment.",
        examples=[2],
    )
    ca: int | None = Field(
        default=None,
        ge=0,
        le=3,
        description="Number of major vessels coloured by fluoroscopy.",
        examples=[0],
    )
    thal: int | None = Field(
        default=None,
        description="Thalassemia result category.",
        examples=[7],
    )
    model_config = {
        "extra": "forbid",
    }

    @field_validator("thal")
    @classmethod
    def validate_thal(cls, value: int | None) -> int | None:
        """Validate known thal categories while permitting missing input."""

        if value is not None and value not in {3, 6, 7}:
            raise ValueError("thal must be one of 3, 6, 7, or null")

        return value


class PredictionResponse(BaseModel):
    """Response returned after model inference."""

    prediction: int
    risk_class: str
    confidence: float
    probability_no_disease: float
    probability_disease: float
    model_name: str
    disclaimer: str


class HealthResponse(BaseModel):
    """Service health-check response."""

    status: str
    model_loaded: bool
    model_name: str
