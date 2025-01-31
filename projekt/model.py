import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import r2_score, mean_squared_error 
from sklearn.preprocessing import StandardScaler 
from statsmodels.stats.outliers_influence import variance_inflation_factor 

def calculate_vif(X, threshold=5.0):
    """
    Calculate VIF for each feature and return features to keep based on threshold.
    Returns columns to keep (with VIF < threshold)
    """
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                       for i in range(X.shape[1])]
    
    return list(vif_data[vif_data["VIF"] < threshold]["Feature"])

def remove_low_variance(X, threshold=0.01):
    """
    Remove columns with variance below threshold
    """
    variances = X.var()
    return list(variances[variances > threshold].index)

def remove_duplicate_columns(X, tolerance=0.0001):
    """
    Remove highly correlated features (correlation > 0.95)
    """
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    return list(set(X.columns) - set(to_drop))

def clean_dataset(X, variance_threshold=0.01):
    """
    Clean dataset using variance and correlation methods
    """
    print(f"Initial number of features: {X.shape[1]}")
    
    # Remove low variance features
    keep_variance = remove_low_variance(X, variance_threshold)
    X = X[keep_variance]
    print(f"Features after removing low variance: {X.shape[1]}")
    
    # Remove highly correlated features
    keep_unique = remove_duplicate_columns(X)
    X = X[keep_unique]
    print(f"Features after removing highly correlated features: {X.shape[1]}")
    
    return X

def train_and_evaluate_model(X, y, test_size=0.2, random_state=42):
    """
    Train RandomForest model and evaluate performance
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\nModel Performance:")
    print(f"R² Score: {r2:.3f}")
    print(f"RMSE: {rmse:.3f}")
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    return model, scaler

# Example usage
if __name__ == "__main__":
    # Load your data
    df = pd.read_csv('add.csv')


    columns_to_drop = ['ChEMBL_ID', 'SMILES', 'pIC50']
    X = df.drop(columns_to_drop, axis=1)
    y = df['pIC50']

    # 2. Convert all columns to numeric, handling any remaining non-numeric values
    X = X.apply(pd.to_numeric, errors='coerce')

    # 3. Remove any columns that have NaN values (in case some couldn't be converted to numeric)
    X = X.dropna(axis=1)
    
    # Clean the dataset
    X_cleaned = clean_dataset(X)
    
    # Train and evaluate model
    model, scaler = train_and_evaluate_model(X_cleaned, y)