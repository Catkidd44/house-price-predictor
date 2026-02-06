"""
Model training script for house price prediction.

Trains multiple models and saves the best performer.
"""

import sys
sys.path.append('..')

import pandas as pd
import numpy as np
import joblib
from datetime import datetime

from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

from data.preprocess import load_and_preprocess_data


def evaluate_model(model, X, y, model_name="Model", cv_folds=5):
    """
    Evaluate model using cross-validation.

    Args:
        model: Scikit-learn model
        X: Features
        y: Target (log-transformed)
        model_name: Name for display
        cv_folds: Number of CV folds

    Returns:
        dict: Evaluation metrics
    """
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # Cross-validation scores (negative MSE)
    cv_scores = cross_val_score(model, X, y, cv=kfold,
                                 scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores)

    # Train on full data for R² score
    model.fit(X, y)
    y_pred = model.predict(X)

    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    results = {
        'model_name': model_name,
        'cv_rmse_mean': cv_rmse.mean(),
        'cv_rmse_std': cv_rmse.std(),
        'train_rmse': rmse,
        'train_mae': mae,
        'train_r2': r2,
        'model': model
    }

    print(f"\n{model_name}:")
    print(f"  CV RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std():.4f})")
    print(f"  Train RMSE: {rmse:.4f}")
    print(f"  Train MAE: {mae:.4f}")
    print(f"  Train R²: {r2:.4f}")

    return results


def train_all_models(X_train, y_train):
    """
    Train multiple models and compare performance.

    Args:
        X_train: Training features
        y_train: Training target (log-transformed)

    Returns:
        dict: Results for all models
    """
    print("="*70)
    print("TRAINING MODELS")
    print("="*70)

    results = {}

    # 1. Linear Regression (Baseline)
    print("\n[1/6] Training Linear Regression...")
    lr = LinearRegression()
    results['linear_regression'] = evaluate_model(lr, X_train, y_train, "Linear Regression")

    # 2. Ridge Regression
    print("\n[2/6] Training Ridge Regression...")
    ridge = Ridge(alpha=10.0, random_state=42)
    results['ridge'] = evaluate_model(ridge, X_train, y_train, "Ridge Regression")

    # 3. Lasso Regression
    print("\n[3/6] Training Lasso Regression...")
    lasso = Lasso(alpha=0.0005, random_state=42, max_iter=10000)
    results['lasso'] = evaluate_model(lasso, X_train, y_train, "Lasso Regression")

    # 4. Random Forest
    print("\n[4/6] Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    results['random_forest'] = evaluate_model(rf, X_train, y_train, "Random Forest")

    # 5. Gradient Boosting
    print("\n[5/6] Training Gradient Boosting...")
    gb = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    results['gradient_boosting'] = evaluate_model(gb, X_train, y_train, "Gradient Boosting")

    # 6. XGBoost
    print("\n[6/6] Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        n_jobs=-1
    )
    results['xgboost'] = evaluate_model(xgb_model, X_train, y_train, "XGBoost")

    return results


def select_best_model(results):
    """
    Select the best model based on CV RMSE.

    Args:
        results: Dictionary of model results

    Returns:
        Best model and its results
    """
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)

    # Create comparison dataframe
    comparison = []
    for model_key, result in results.items():
        comparison.append({
            'Model': result['model_name'],
            'CV RMSE': result['cv_rmse_mean'],
            'CV Std': result['cv_rmse_std'],
            'Train RMSE': result['train_rmse'],
            'Train R²': result['train_r2']
        })

    comparison_df = pd.DataFrame(comparison)
    comparison_df = comparison_df.sort_values('CV RMSE')

    print("\n" + comparison_df.to_string(index=False))

    # Select best model (lowest CV RMSE)
    best_model_key = min(results.keys(), key=lambda k: results[k]['cv_rmse_mean'])
    best_result = results[best_model_key]

    print(f"\n{'='*70}")
    print(f"BEST MODEL: {best_result['model_name']}")
    print(f"  CV RMSE: {best_result['cv_rmse_mean']:.4f}")
    print(f"  Train R²: {best_result['train_r2']:.4f}")
    print(f"{'='*70}")

    return best_result['model'], best_result


def save_model(model, preprocessor, metadata, model_dir='../../models'):
    """
    Save the trained model and preprocessor.

    Args:
        model: Trained model
        preprocessor: Fitted preprocessor
        metadata: Model metadata dict
        model_dir: Directory to save models
    """
    import os
    os.makedirs(model_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save model
    model_path = f"{model_dir}/model_v1.pkl"
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")

    # Save preprocessor
    preprocessor_path = f"{model_dir}/preprocessor.pkl"
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Preprocessor saved to: {preprocessor_path}")

    # Save metadata
    metadata_df = pd.DataFrame([metadata])
    metadata_path = f"{model_dir}/model_metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)
    print(f"Metadata saved to: {metadata_path}")

    print("\nAll model artifacts saved successfully!")


def main():
    """Main training pipeline."""
    print("="*70)
    print("HOUSE PRICE PREDICTION - MODEL TRAINING")
    print("="*70)

    # Load and preprocess data
    print("\nStep 1: Loading and preprocessing data...")
    X_train, y_train, preprocessor = load_and_preprocess_data(
        '../../data/raw/train.csv'
    )

    print(f"\nFinal data shape: X={X_train.shape}, y={y_train.shape}")

    # Train models
    print("\nStep 2: Training models...")
    results = train_all_models(X_train, y_train)

    # Select best model
    print("\nStep 3: Selecting best model...")
    best_model, best_result = select_best_model(results)

    # Save model
    print("\nStep 4: Saving model...")
    metadata = {
        'model_name': best_result['model_name'],
        'cv_rmse': best_result['cv_rmse_mean'],
        'cv_rmse_std': best_result['cv_rmse_std'],
        'train_rmse': best_result['train_rmse'],
        'train_mae': best_result['train_mae'],
        'train_r2': best_result['train_r2'],
        'n_features': X_train.shape[1],
        'n_samples': X_train.shape[0],
        'trained_on': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    save_model(best_model, preprocessor, metadata)

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nBest Model: {best_result['model_name']}")
    print(f"Expected RMSE on unseen data: ~{best_result['cv_rmse_mean']:.4f} (log scale)")
    print(f"This translates to ~${np.expm1(best_result['cv_rmse_mean']):,.0f} in actual prices")
    print("\nNext steps:")
    print("  1. Test model on validation data")
    print("  2. Build API (Phase 4)")
    print("  3. Create Docker container (Phase 5)")
    print("="*70)


if __name__ == "__main__":
    main()
