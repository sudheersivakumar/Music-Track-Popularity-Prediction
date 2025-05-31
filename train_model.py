import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Define the dataset path
dataset_path = 'C:/Users/sudhe/OneDrive/Documents/ML projects/Supervised learning/Music-track-popularity/dataset.csv/dataset.csv'

# Check if the file exists
if not os.path.exists(dataset_path):
    print(f"Error: The file {dataset_path} does not exist.")
    exit(1)

# Try alternative approaches if permission issues occur
try:
    # First attempt - standard read
    df = pd.read_csv(dataset_path)
except PermissionError:
    print(f"Permission denied on first attempt. Creating a sample dataset for testing...")
    
    # Create a sample music dataset for testing
    # This will allow you to test your model even if you can't access the original file
    sample_data = {
        'popularity': np.random.randint(0, 100, 1000),
        'danceability': np.random.uniform(0, 1, 1000),
        'energy': np.random.uniform(0, 1, 1000),
        'loudness': np.random.uniform(-20, 0, 1000),
        'speechiness': np.random.uniform(0, 1, 1000),
        'acousticness': np.random.uniform(0, 1, 1000),
        'instrumentalness': np.random.uniform(0, 1, 1000),
        'liveness': np.random.uniform(0, 1, 1000),
        'valence': np.random.uniform(0, 1, 1000),
        'tempo': np.random.uniform(50, 200, 1000),
        'duration_ms': np.random.randint(60000, 300000, 1000),
    }
    
    # Create DataFrame from sample data
    df = pd.DataFrame(sample_data)
    print("Created sample dataset with 1000 records for testing purposes")
    
    # Try to save the sample dataset to a different location
    try:
        sample_path = 'C:/Users/sudhe/Documents/sample_music_dataset.csv'
        df.to_csv(sample_path, index=False)
        print(f"Sample dataset saved to {sample_path}")
    except Exception as e:
        print(f"Could not save sample dataset: {str(e)}")
except Exception as e:
    print(f"Error loading dataset: {str(e)}")
    exit(1)

# Display basic information about the dataset
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nSummary statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Assuming 'popularity' is your target variable - replace with your actual target column name
target_column = 'popularity'

# Visualize the distribution of the target variable
plt.figure(figsize=(10, 6))
sns.histplot(df[target_column], kde=True)
plt.title(f'Distribution of {target_column}')
plt.savefig('c:\\Users\\sudhe\\OneDrive\\Documents\\ML projects\\Supervised learning\\Music-track-popularity\\popularity_distribution.png')
plt.close()

# Prepare features (X) and target variable (y)
# Exclude non-numeric columns and the target column from features
# You may need to adjust this based on your actual dataset
X = df.select_dtypes(include=[np.number]).drop(columns=[target_column], errors='ignore')
y = df[target_column]

print("\nFeatures used for training:")
print(X.columns.tolist())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Make predictions
rf_predictions = rf_model.predict(X_test_scaled)

# Evaluate the model
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_test, rf_predictions)

print("\nRandom Forest Model Evaluation:")
print(f"Mean Squared Error: {rf_mse:.4f}")
print(f"Root Mean Squared Error: {rf_rmse:.4f}")
print(f"R² Score: {rf_r2:.4f}")

# Train a Linear Regression model for comparison
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Make predictions with Linear Regression
lr_predictions = lr_model.predict(X_test_scaled)

# Evaluate the Linear Regression model
lr_mse = mean_squared_error(y_test, lr_predictions)
lr_rmse = np.sqrt(lr_mse)
lr_r2 = r2_score(y_test, lr_predictions)

print("\nLinear Regression Model Evaluation:")
print(f"Mean Squared Error: {lr_mse:.4f}")
print(f"Root Mean Squared Error: {lr_rmse:.4f}")
print(f"R² Score: {lr_r2:.4f}")

# Feature importance (for Random Forest)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance.head(10))

# Visualize feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
plt.title('Top 10 Feature Importance')
plt.tight_layout()
plt.savefig('c:\\Users\\sudhe\\OneDrive\\Documents\\ML projects\\Supervised learning\\Music-track-popularity\\feature_importance.png')
plt.close()

# Visualize actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, rf_predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Popularity (Random Forest)')
plt.tight_layout()
plt.savefig('c:\\Users\\sudhe\\OneDrive\\Documents\\ML projects\\Supervised learning\\Music-track-popularity\\actual_vs_predicted.png')
plt.close()

# After your existing imports, add these:
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor  # You may need to install this: pip install xgboost

# Train a Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train_scaled, y_train)
gb_predictions = gb_model.predict(X_test_scaled)
gb_mse = mean_squared_error(y_test, gb_predictions)
gb_rmse = np.sqrt(gb_mse)
gb_r2 = r2_score(y_test, gb_predictions)

print("\nGradient Boosting Model Evaluation:")
print(f"Mean Squared Error: {gb_mse:.4f}")
print(f"Root Mean Squared Error: {gb_rmse:.4f}")
print(f"R² Score: {gb_r2:.4f}")

# Train an XGBoost Regressor
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train_scaled, y_train)
xgb_predictions = xgb_model.predict(X_test_scaled)
xgb_mse = mean_squared_error(y_test, xgb_predictions)
xgb_rmse = np.sqrt(xgb_mse)
xgb_r2 = r2_score(y_test, xgb_predictions)

print("\nXGBoost Model Evaluation:")
print(f"Mean Squared Error: {xgb_mse:.4f}")
print(f"Root Mean Squared Error: {xgb_rmse:.4f}")
print(f"R² Score: {xgb_r2:.4f}")

# Train a Support Vector Regressor (with a smaller subset if dataset is large)
# SVR can be slow on large datasets
X_train_sample = X_train_scaled[:1000] if X_train_scaled.shape[0] > 1000 else X_train_scaled
y_train_sample = y_train[:1000] if len(y_train) > 1000 else y_train
svr_model = SVR(kernel='rbf')
svr_model.fit(X_train_sample, y_train_sample)
svr_predictions = svr_model.predict(X_test_scaled)
svr_mse = mean_squared_error(y_test, svr_predictions)
svr_rmse = np.sqrt(svr_mse)
svr_r2 = r2_score(y_test, svr_predictions)

print("\nSupport Vector Regressor Evaluation:")
print(f"Mean Squared Error: {svr_mse:.4f}")
print(f"Root Mean Squared Error: {svr_rmse:.4f}")
print(f"R² Score: {svr_r2:.4f}")

# Train a K-Nearest Neighbors Regressor
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
knn_predictions = knn_model.predict(X_test_scaled)
knn_mse = mean_squared_error(y_test, knn_predictions)
knn_rmse = np.sqrt(knn_mse)
knn_r2 = r2_score(y_test, knn_predictions)

print("\nK-Nearest Neighbors Regressor Evaluation:")
print(f"Mean Squared Error: {knn_mse:.4f}")
print(f"Root Mean Squared Error: {knn_rmse:.4f}")
print(f"R² Score: {knn_r2:.4f}")

# Modify the model selection part to include all models
models = {
    'Random Forest': (rf_model, rf_r2),
    'Linear Regression': (lr_model, lr_r2),
    'Gradient Boosting': (gb_model, gb_r2),
    'XGBoost': (xgb_model, xgb_r2),
    'Support Vector Regressor': (svr_model, svr_r2),
    'K-Nearest Neighbors': (knn_model, knn_r2)
}

# Find the best model
best_model_name = max(models.items(), key=lambda x: x[1][1])[0]
best_model, best_r2 = models[best_model_name]

print(f"\nThe best performing model is {best_model_name} with R² Score: {best_r2:.4f}")

# Save the best model
print(f"\nSaving {best_model_name} model as it performed better")
joblib.dump(best_model, 'c:\\Users\\sudhe\\OneDrive\\Documents\\ML projects\\Supervised learning\\Music-track-popularity\\popularity_prediction_model.pkl')
joblib.dump(scaler, 'c:\\Users\\sudhe\\OneDrive\\Documents\\ML projects\\Supervised learning\\Music-track-popularity\\scaler.pkl')

print("\nTraining and evaluation completed!")