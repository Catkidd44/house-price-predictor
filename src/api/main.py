"""
FastAPI application for house price prediction.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from api.schemas import (
    HouseFeatures,
    SimplifiedHouseFeatures,
    PredictionResponse,
    HealthResponse
)

# Initialize FastAPI app
app = FastAPI(
    title="House Price Predictor API",
    description="Predict house prices in Ames, Iowa using machine learning",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = Path(__file__).parent.parent.parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Global variables for model and preprocessor
model = None
preprocessor = None
model_metadata = None


def load_model_artifacts():
    """Load trained model and preprocessor."""
    global model, preprocessor, model_metadata

    try:
        model_dir = Path(__file__).parent.parent.parent / "models"

        # Load model
        model_path = model_dir / "model_v1.pkl"
        model = joblib.load(model_path)
        print(f"[OK] Model loaded from {model_path}")

        # Load preprocessor
        preprocessor_path = model_dir / "preprocessor.pkl"
        preprocessor = joblib.load(preprocessor_path)
        print(f"[OK] Preprocessor loaded from {preprocessor_path}")

        # Load metadata
        metadata_path = model_dir / "model_metadata.csv"
        if metadata_path.exists():
            model_metadata = pd.read_csv(metadata_path).iloc[0].to_dict()
            print(f"[OK] Metadata loaded: {model_metadata['model_name']}")

        return True
    except Exception as e:
        print(f"[ERROR] Error loading model artifacts: {e}")
        return False


# Load model on startup
@app.on_event("startup")
async def startup_event():
    """Load model when API starts."""
    success = load_model_artifacts()
    if success:
        print("="*50)
        print("House Price Predictor API is ready!")
        print("="*50)
    else:
        print("WARNING: Model not loaded. Predictions will fail.")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page."""
    html_path = Path(__file__).parent.parent.parent / "static" / "index.html"

    if html_path.exists():
        with open(html_path, 'r') as f:
            return f.read()
    else:
        return """
        <html>
            <body>
                <h1>House Price Predictor API</h1>
                <p>Frontend not found. Visit <a href="/docs">/docs</a> for API documentation.</p>
            </body>
        </html>
        """


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_name=model_metadata.get('model_name') if model_metadata else None,
        version="1.0.0"
    )


def create_full_feature_dict(input_features: dict) -> dict:
    """
    Create a complete feature dictionary with defaults for missing features.

    This is a simplified version - in production, you'd want to handle this more robustly.
    """
    # Default values for all features (based on training data)
    defaults = {
        'MSSubClass': 60,
        'MSZoning': input_features.get('MSZoning', 'RL'),
        'LotFrontage': 68.0,
        'LotArea': input_features.get('LotArea', 8000),
        'Street': 'Pave',
        'LotShape': 'Reg',
        'LandContour': 'Lvl',
        'Utilities': 'AllPub',
        'LotConfig': 'Inside',
        'LandSlope': 'Gtl',
        'Neighborhood': input_features.get('Neighborhood', 'NAmes'),
        'Condition1': 'Norm',
        'Condition2': 'Norm',
        'BldgType': '1Fam',
        'HouseStyle': '1Story',
        'OverallQual': input_features.get('OverallQual'),
        'OverallCond': 5,
        'YearBuilt': input_features.get('YearBuilt'),
        'YearRemodAdd': input_features.get('YearRemodAdd'),
        'RoofStyle': 'Gable',
        'RoofMatl': 'CompShg',
        'Exterior1st': 'VinylSd',
        'Exterior2nd': 'VinylSd',
        'MasVnrType': 'None',
        'MasVnrArea': 0,
        'ExterQual': 'TA',
        'ExterCond': 'TA',
        'Foundation': 'PConc',
        'BsmtQual': 'TA',
        'BsmtCond': 'TA',
        'BsmtExposure': 'No',
        'BsmtFinType1': 'GLQ',
        'BsmtFinSF1': 500,
        'BsmtFinType2': 'Unf',
        'BsmtFinSF2': 0,
        'BsmtUnfSF': 500,
        'TotalBsmtSF': input_features.get('TotalBsmtSF'),
        'Heating': 'GasA',
        'HeatingQC': 'Ex',
        'CentralAir': 'Y',
        'Electrical': 'SBrkr',
        '1stFlrSF': input_features.get('FirstFlrSF', input_features.get('1stFlrSF')),
        '2ndFlrSF': input_features.get('SecondFlrSF', input_features.get('2ndFlrSF', 0)),
        'LowQualFinSF': 0,
        'GrLivArea': input_features.get('GrLivArea'),
        'BsmtFullBath': input_features.get('BsmtFullBath', 0),
        'BsmtHalfBath': 0,
        'FullBath': input_features.get('FullBath'),
        'HalfBath': input_features.get('HalfBath', 0),
        'BedroomAbvGr': input_features.get('BedroomAbvGr', 3),
        'KitchenAbvGr': input_features.get('KitchenAbvGr', 1),
        'KitchenQual': 'TA',
        'TotRmsAbvGrd': input_features.get('TotRmsAbvGrd', 6),
        'Functional': 'Typ',
        'Fireplaces': input_features.get('Fireplaces', 0),
        'FireplaceQu': 'None',
        'GarageType': 'Attchd',
        'GarageYrBlt': input_features.get('GarageYrBlt', input_features.get('YearBuilt')),
        'GarageFinish': 'RFn',
        'GarageCars': input_features.get('GarageCars'),
        'GarageArea': input_features.get('GarageArea', 500),
        'GarageQual': 'TA',
        'GarageCond': 'TA',
        'PavedDrive': 'Y',
        'WoodDeckSF': 0,
        'OpenPorchSF': 0,
        'EnclosedPorch': 0,
        '3SsnPorch': 0,
        'ScreenPorch': 0,
        'PoolArea': 0,
        'MiscVal': 0,
        'MoSold': 6,
        'YrSold': 2010,
        'SaleType': 'WD',
        'SaleCondition': 'Normal'
    }

    return defaults


@app.post("/predict", response_model=PredictionResponse)
async def predict_price(features: HouseFeatures):
    """
    Predict house price based on input features.

    Returns predicted price in USD.
    """
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please restart the API."
        )

    try:
        # Convert Pydantic model to dict
        input_dict = features.model_dump(by_alias=True)

        # Create full feature dictionary with defaults
        full_features = create_full_feature_dict(input_dict)

        # Convert to DataFrame (single row)
        input_df = pd.DataFrame([full_features])

        # Preprocess using fitted preprocessor
        # The preprocessor handles all feature engineering and encoding
        X_processed = preprocessor.transform(input_df)

        # Make prediction (log scale)
        log_prediction = model.predict(X_processed)[0]

        # Convert back to actual price
        predicted_price = np.expm1(log_prediction)

        # Format price
        formatted_price = f"${predicted_price:,.0f}"

        # Estimate confidence interval (simple ±15% for demo)
        lower_bound = predicted_price * 0.85
        upper_bound = predicted_price * 1.15

        return PredictionResponse(
            predicted_price=float(predicted_price),
            predicted_price_formatted=formatted_price,
            model_name=model_metadata.get('model_name', 'Unknown') if model_metadata else 'Unknown',
            confidence_interval={
                "lower": float(lower_bound),
                "upper": float(upper_bound),
                "lower_formatted": f"${lower_bound:,.0f}",
                "upper_formatted": f"${upper_bound:,.0f}"
            }
        )

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Prediction error: {error_details}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/simple", response_model=PredictionResponse)
async def predict_price_simple(features: SimplifiedHouseFeatures):
    """
    Predict house price using simplified input (top features only).

    This endpoint uses default values for non-essential features.
    """
    # Convert to HouseFeatures with defaults
    full_features = HouseFeatures(
        OverallQual=features.OverallQual,
        GrLivArea=features.GrLivArea,
        GarageCars=features.GarageCars,
        GarageArea=features.GarageCars * 250,  # Estimate
        TotalBsmtSF=features.TotalBsmtSF,
        FirstFlrSF=features.FirstFlrSF,
        FullBath=features.FullBath,
        TotRmsAbvGrd=features.GrLivArea // 250,  # Estimate
        YearBuilt=features.YearBuilt,
        YearRemodAdd=features.YearRemodAdd
    )

    return await predict_price(full_features)


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model."""
    if model_metadata is None:
        raise HTTPException(status_code=503, detail="Model metadata not available")

    return {
        "model_name": model_metadata.get('model_name'),
        "cv_rmse": model_metadata.get('cv_rmse'),
        "train_r2": model_metadata.get('train_r2'),
        "n_features": model_metadata.get('n_features'),
        "n_samples": model_metadata.get('n_samples'),
        "trained_on": model_metadata.get('trained_on')
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
