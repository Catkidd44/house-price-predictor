"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional


class HouseFeatures(BaseModel):
    """Input features for house price prediction."""

    # Most important features based on EDA
    OverallQual: int = Field(..., ge=1, le=10, description="Overall material and finish quality (1-10)")
    GrLivArea: int = Field(..., gt=0, description="Above grade living area (sq ft)")
    GarageCars: int = Field(..., ge=0, le=5, description="Size of garage in car capacity")
    GarageArea: int = Field(..., ge=0, description="Size of garage in square feet")
    TotalBsmtSF: int = Field(..., ge=0, description="Total square feet of basement area")
    FirstFlrSF: int = Field(..., gt=0, alias="1stFlrSF", description="First floor square feet")
    FullBath: int = Field(..., ge=0, le=5, description="Full bathrooms above grade")
    TotRmsAbvGrd: int = Field(..., ge=0, description="Total rooms above grade")
    YearBuilt: int = Field(..., ge=1800, le=2025, description="Original construction date")
    YearRemodAdd: int = Field(..., ge=1800, le=2025, description="Remodel date")

    # Additional important features
    LotArea: int = Field(default=8000, ge=0, description="Lot size in square feet")
    SecondFlrSF: int = Field(default=0, ge=0, alias="2ndFlrSF", description="Second floor square feet")
    BsmtFullBath: int = Field(default=0, ge=0, description="Basement full bathrooms")
    HalfBath: int = Field(default=0, ge=0, description="Half baths above grade")
    BedroomAbvGr: int = Field(default=3, ge=0, description="Bedrooms above grade")
    KitchenAbvGr: int = Field(default=1, ge=0, description="Kitchens above grade")
    Fireplaces: int = Field(default=0, ge=0, description="Number of fireplaces")
    GarageYrBlt: Optional[int] = Field(default=None, ge=1800, le=2025, description="Year garage was built")

    # Categorical - simplified for demo
    Neighborhood: str = Field(default="NAmes", description="Physical location within Ames city limits")
    MSZoning: str = Field(default="RL", description="General zoning classification")

    class Config:
        json_schema_extra = {
            "example": {
                "OverallQual": 7,
                "GrLivArea": 1500,
                "GarageCars": 2,
                "GarageArea": 500,
                "TotalBsmtSF": 1000,
                "1stFlrSF": 1000,
                "FullBath": 2,
                "TotRmsAbvGrd": 7,
                "YearBuilt": 2000,
                "YearRemodAdd": 2000,
                "LotArea": 8000,
                "2ndFlrSF": 500,
                "BsmtFullBath": 1,
                "HalfBath": 0,
                "BedroomAbvGr": 3,
                "KitchenAbvGr": 1,
                "Fireplaces": 1,
                "GarageYrBlt": 2000,
                "Neighborhood": "NAmes",
                "MSZoning": "RL"
            }
        }


class SimplifiedHouseFeatures(BaseModel):
    """Simplified input for quick predictions (top 10 features only)."""

    OverallQual: int = Field(..., ge=1, le=10, description="Overall quality (1-10)")
    GrLivArea: int = Field(..., gt=0, description="Living area (sq ft)")
    GarageCars: int = Field(..., ge=0, le=5, description="Garage size (cars)")
    TotalBsmtSF: int = Field(..., ge=0, description="Basement area (sq ft)")
    FirstFlrSF: int = Field(..., gt=0, alias="1stFlrSF", description="First floor (sq ft)")
    FullBath: int = Field(..., ge=0, le=5, description="Full bathrooms")
    YearBuilt: int = Field(..., ge=1800, le=2025, description="Year built")
    YearRemodAdd: int = Field(..., ge=1800, le=2025, description="Year remodeled")

    class Config:
        json_schema_extra = {
            "example": {
                "OverallQual": 7,
                "GrLivArea": 1500,
                "GarageCars": 2,
                "TotalBsmtSF": 1000,
                "1stFlrSF": 1000,
                "FullBath": 2,
                "YearBuilt": 2000,
                "YearRemodAdd": 2000
            }
        }


class PredictionResponse(BaseModel):
    """Response containing predicted house price."""

    predicted_price: float = Field(..., description="Predicted house price in USD")
    predicted_price_formatted: str = Field(..., description="Formatted price with currency")
    model_name: str = Field(..., description="Model used for prediction")
    confidence_interval: Optional[dict] = Field(None, description="Prediction confidence interval")

    class Config:
        json_schema_extra = {
            "example": {
                "predicted_price": 180000.50,
                "predicted_price_formatted": "$180,000",
                "model_name": "Lasso Regression",
                "confidence_interval": {
                    "lower": 160000,
                    "upper": 200000
                }
            }
        }


class HealthResponse(BaseModel):
    """API health check response."""

    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_name: Optional[str] = Field(None, description="Loaded model name")
    version: str = Field(..., description="API version")
