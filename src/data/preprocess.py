"""
Data preprocessing module for house price prediction.

This module handles:
- Missing value imputation
- Feature engineering
- Outlier removal
- Categorical encoding
- Feature scaling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class HousePricePreprocessor:
    """Preprocessor for house price prediction data."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.features_to_drop = ['Id', 'PoolQC', 'MiscFeature', 'Alley', 'Fence']
        self.feature_names = None  # Store feature names after fit

    def remove_outliers(self, df, is_train=True):
        """Remove outliers based on EDA findings."""
        if is_train and 'SalePrice' in df.columns:
            # Remove houses with large living area but low price
            df = df.drop(df[(df['GrLivArea'] > 4000) & (df['SalePrice'] < 200000)].index)
        return df

    def handle_missing_values(self, df):
        """Impute missing values based on feature type."""
        df = df.copy()

        # Drop features with >80% missing
        df = df.drop(columns=[col for col in self.features_to_drop if col in df.columns])

        # LotFrontage: fill with median by Neighborhood
        if 'LotFrontage' in df.columns:
            df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
                lambda x: x.fillna(x.median())
            )

        # Garage features
        garage_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
        for col in garage_cols:
            if col in df.columns:
                df[col] = df[col].fillna('None')

        if 'GarageYrBlt' in df.columns:
            df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)

        if 'GarageArea' in df.columns:
            df['GarageArea'] = df['GarageArea'].fillna(0)

        if 'GarageCars' in df.columns:
            df['GarageCars'] = df['GarageCars'].fillna(0)

        # Basement features
        basement_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
        for col in basement_cols:
            if col in df.columns:
                df[col] = df[col].fillna('None')

        basement_num_cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']
        for col in basement_num_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # MasVnrType and MasVnrArea
        if 'MasVnrType' in df.columns:
            df['MasVnrType'] = df['MasVnrType'].fillna('None')
        if 'MasVnrArea' in df.columns:
            df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

        # MSZoning (mode)
        if 'MSZoning' in df.columns:
            df['MSZoning'] = df['MSZoning'].fillna(df['MSZoning'].mode()[0])

        # Utilities (mode)
        if 'Utilities' in df.columns:
            df['Utilities'] = df['Utilities'].fillna(df['Utilities'].mode()[0])

        # Functional (mode)
        if 'Functional' in df.columns:
            df['Functional'] = df['Functional'].fillna('Typ')

        # Electrical (mode)
        if 'Electrical' in df.columns:
            df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

        # KitchenQual (mode)
        if 'KitchenQual' in df.columns:
            df['KitchenQual'] = df['KitchenQual'].fillna(df['KitchenQual'].mode()[0])

        # Exterior (mode)
        if 'Exterior1st' in df.columns:
            df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])
        if 'Exterior2nd' in df.columns:
            df['Exterior2nd'] = df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0])

        # SaleType (mode)
        if 'SaleType' in df.columns:
            df['SaleType'] = df['SaleType'].fillna(df['SaleType'].mode()[0])

        # FireplaceQu
        if 'FireplaceQu' in df.columns:
            df['FireplaceQu'] = df['FireplaceQu'].fillna('None')

        return df

    def create_features(self, df):
        """Engineer new features."""
        df = df.copy()

        # Total square footage
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

        # House age
        if 'YrSold' in df.columns and 'YearBuilt' in df.columns:
            df['Age'] = df['YrSold'] - df['YearBuilt']

        # Remodeling indicator
        if 'YearRemodAdd' in df.columns and 'YearBuilt' in df.columns:
            df['IsRemodeled'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)

        # Total bathrooms
        df['TotalBath'] = df['FullBath'] + 0.5 * df['HalfBath'] + df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']

        # Total porch area
        porch_cols = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
        df['TotalPorchSF'] = df[porch_cols].sum(axis=1)

        # Has pool/garage/basement/fireplace
        if 'PoolArea' in df.columns:
            df['HasPool'] = (df['PoolArea'] > 0).astype(int)
        if 'GarageArea' in df.columns:
            df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
        if 'TotalBsmtSF' in df.columns:
            df['HasBsmt'] = (df['TotalBsmtSF'] > 0).astype(int)
        if 'Fireplaces' in df.columns:
            df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)

        return df

    def encode_categorical(self, df, is_train=True):
        """Encode categorical variables."""
        df = df.copy()

        # Ordinal features (quality/condition ratings)
        quality_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
        quality_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
                       'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond']

        for col in quality_cols:
            if col in df.columns:
                df[col] = df[col].map(quality_map)

        # BsmtExposure
        if 'BsmtExposure' in df.columns:
            bsmt_exposure_map = {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
            df['BsmtExposure'] = df['BsmtExposure'].map(bsmt_exposure_map)

        # BsmtFinType
        bsmt_fin_map = {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
        for col in ['BsmtFinType1', 'BsmtFinType2']:
            if col in df.columns:
                df[col] = df[col].map(bsmt_fin_map)

        # GarageFinish
        if 'GarageFinish' in df.columns:
            garage_finish_map = {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
            df['GarageFinish'] = df['GarageFinish'].map(garage_finish_map)

        # Get categorical columns for one-hot encoding
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        # One-hot encode remaining categorical features
        if categorical_cols:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        return df

    def scale_features(self, X, is_train=True):
        """Scale numerical features."""
        column_names = X.columns.tolist()

        if is_train:
            self.feature_names = column_names  # Save feature names
            X_scaled = self.scaler.fit_transform(X.values)  # Use .values to avoid feature name check
        else:
            X_scaled = self.scaler.transform(X.values)  # Use .values to avoid feature name check

        return pd.DataFrame(X_scaled, columns=column_names, index=X.index)

    def fit_transform(self, df, target_col='SalePrice'):
        """Complete preprocessing pipeline for training data."""
        print("Starting preprocessing...")

        # Remove outliers
        df = self.remove_outliers(df, is_train=True)
        print(f"After outlier removal: {len(df)} samples")

        # Separate target
        if target_col in df.columns:
            y = df[target_col].copy()
            X = df.drop(columns=[target_col])
        else:
            y = None
            X = df.copy()

        # Handle missing values
        X = self.handle_missing_values(X)
        print(f"Missing values handled. Remaining columns: {len(X.columns)}")

        # Create new features
        X = self.create_features(X)
        print(f"After feature engineering: {len(X.columns)} features")

        # Encode categorical
        X = self.encode_categorical(X, is_train=True)
        print(f"After encoding: {len(X.columns)} features")

        # Scale features
        X = self.scale_features(X, is_train=True)
        print(f"Features scaled")

        print(f"Preprocessing complete! Final shape: {X.shape}")

        return X, y

    def transform(self, df):
        """Transform test data using fitted preprocessor."""
        print("Transforming test data...")

        # Handle missing values
        X = self.handle_missing_values(df)

        # Create new features
        X = self.create_features(X)

        # Encode categorical
        X = self.encode_categorical(X, is_train=False)

        # Align columns with training data
        if self.feature_names:
            # Add missing columns with 0
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0
            # Keep only training columns in same order
            X = X[self.feature_names]

        # Scale features
        X = self.scale_features(X, is_train=False)

        print(f"Test data transformed! Shape: {X.shape}")

        return X


def load_and_preprocess_data(train_path, test_path=None):
    """
    Load and preprocess training (and optionally test) data.

    Args:
        train_path: Path to training CSV
        test_path: Path to test CSV (optional)

    Returns:
        X_train, y_train, X_test (if test_path provided)
    """
    # Load data
    train_df = pd.read_csv(train_path)
    print(f"Loaded training data: {train_df.shape}")

    # Initialize preprocessor
    preprocessor = HousePricePreprocessor()

    # Preprocess training data
    X_train, y_train = preprocessor.fit_transform(train_df)

    # Apply log transformation to target
    y_train_log = np.log1p(y_train)
    print(f"Target transformed with log1p. Skewness reduced from {y_train.skew():.3f} to {y_train_log.skew():.3f}")

    if test_path:
        test_df = pd.read_csv(test_path)
        print(f"\nLoaded test data: {test_df.shape}")
        X_test = preprocessor.transform(test_df)
        return X_train, y_train_log, X_test, preprocessor

    return X_train, y_train_log, preprocessor


if __name__ == "__main__":
    # Test preprocessing
    X_train, y_train, preprocessor = load_and_preprocess_data(
        '../../data/raw/train.csv'
    )
    print(f"\nFinal training data shape: X={X_train.shape}, y={y_train.shape}")
