# Music Track Popularity Prediction

A machine learning project that predicts music track popularity using various audio features and multiple regression algorithms.

## Overview

This project implements a comprehensive machine learning pipeline to predict the popularity of music tracks based on their audio characteristics. The system compares multiple regression algorithms and automatically selects the best-performing model based on R² score.

## Features

- **Multiple Algorithm Comparison**: Implements 6 different regression algorithms
- **Automated Model Selection**: Automatically selects the best-performing model
- **Comprehensive Evaluation**: Provides detailed metrics for each model
- **Visualization**: Generates insightful plots for data analysis
- **Robust Error Handling**: Includes fallback mechanisms for data loading issues
- **Feature Importance Analysis**: Identifies the most influential audio features

## Dataset

The project uses audio features to predict track popularity, including:

- **Danceability**: How suitable a track is for dancing
- **Energy**: Perceptual measure of intensity and power
- **Loudness**: Overall loudness of a track in decibels
- **Speechiness**: Presence of spoken words in a track
- **Acousticness**: Confidence measure of whether the track is acoustic
- **Instrumentalness**: Predicts whether a track contains no vocals
- **Liveness**: Detects the presence of an audience in the recording
- **Valence**: Musical positiveness conveyed by a track
- **Tempo**: Overall estimated tempo in beats per minute
- **Duration**: Length of the track in milliseconds

**Target Variable**: `popularity` (0-100 scale)

## Machine Learning Models

The project implements and compares the following algorithms:

1. **Random Forest Regressor** - Ensemble method using multiple decision trees
2. **Linear Regression** - Simple linear relationship modeling
3. **Gradient Boosting Regressor** - Sequential ensemble method
4. **XGBoost Regressor** - Optimized gradient boosting framework
5. **Support Vector Regressor (SVR)** - Non-linear regression with RBF kernel
6. **K-Nearest Neighbors Regressor** - Instance-based learning algorithm

## Installation

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib xgboost
```

### Required Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
```

## Usage

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd music-track-popularity
   ```

2. **Prepare your dataset**
   - Place your dataset CSV file in the project directory
   - Update the `dataset_path` variable in `train_model.py`

3. **Run the training script**
   ```bash
   python train_model.py
   ```

4. **View results**
   - Check console output for model performance metrics
   - Generated visualizations will be saved as PNG files
   - Best model will be saved as `popularity_prediction_model.pkl`

## Project Structure

```
music-track-popularity/
├── train_model.py              # Main training script
├── dataset.csv                 # Your music dataset
├── popularity_prediction_model.pkl  # Saved best model
├── scaler.pkl                  # Saved feature scaler
├── popularity_distribution.png # Target variable distribution
├── feature_importance.png      # Feature importance visualization
├── actual_vs_predicted.png     # Model performance visualization
└── README.md                   # This file
```

## Model Evaluation Metrics

The project evaluates each model using:

- **Mean Squared Error (MSE)**: Average of squared differences between actual and predicted values
- **Root Mean Squared Error (RMSE)**: Square root of MSE, in same units as target variable
- **R² Score**: Coefficient of determination, indicates proportion of variance explained

## Key Features

### Automated Data Handling
- Automatic detection of numeric features
- Robust error handling for file access issues
- Sample data generation for testing purposes

### Comprehensive Model Comparison
- Side-by-side evaluation of multiple algorithms
- Automatic selection of best-performing model
- Detailed performance metrics for each model

### Feature Analysis
- Feature importance ranking (for tree-based models)
- Visualization of most influential audio characteristics
- Data distribution analysis

### Visualization Outputs
- **Popularity Distribution**: Histogram showing target variable distribution
- **Feature Importance**: Bar chart of top 10 most important features
- **Actual vs Predicted**: Scatter plot showing model accuracy

## Results Interpretation

The system automatically selects the best model based on R² score. Typical performance expectations:

- **R² > 0.7**: Excellent model performance
- **R² 0.5-0.7**: Good model performance
- **R² 0.3-0.5**: Moderate model performance
- **R² < 0.3**: Poor model performance

## Customization

### Adding New Features
1. Include additional audio features in your dataset
2. The script automatically detects numeric columns
3. Update feature preprocessing if needed

### Hyperparameter Tuning
```python
# Example: Tuning Random Forest
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
```

### Cross-Validation
```python
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
```

## Troubleshooting

### Common Issues

1. **File Permission Error**
   - The script includes automatic fallback to sample data
   - Check file path and permissions

2. **Missing Libraries**
   ```bash
   pip install --upgrade scikit-learn xgboost
   ```

3. **Memory Issues with Large Datasets**
   - SVR automatically uses subset for training if dataset > 1000 samples
   - Consider reducing dataset size for testing

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Future Enhancements

- [ ] Implement cross-validation for more robust evaluation
- [ ] Add hyperparameter tuning with GridSearchCV
- [ ] Include deep learning models (Neural Networks)
- [ ] Add real-time prediction API
- [ ] Implement feature selection techniques
- [ ] Add model interpretation with SHAP values

## Contact

For questions or suggestions, please open an issue or reach out to the project maintainer.

---

**Note**: This project is designed for educational and research purposes. Model performance may vary based on dataset quality and size.
