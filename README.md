# Movie Score Predictions

## Project Overview

In this project, I collected data from the TMDB website(https://www.themoviedb.org) and evaluated the performance of ten popular Machine Learning models used in Table-Based data analysis to predict the movie score. I also performed feature extraction and wrote a detailed exploratory data analysis.

## Feature Extraction and EDA

Details of the feature extraction process and the results of the EDA can be found [here](notebooks/2_EDA.ipynb).

## Packages Used

The following Python libraries were utilized for data analysis and visualization:
- Pandas
- NumPy
- Matplotlib
- Seaborn

## Models Tested

The following Machine Learning models were tested in this project:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost
- K-Nearest Neighbors (KNN)

## Results

### Dataset Versions

All models were tested on three different versions of the dataset, containing different numbers of features:
1. 4 features (only numeric)
2. 9 features (numeric and encoded categorical features)
3. 17 features (enhancing with more complicated categorical features)

### Model Performance

- Random Forest-based models, including more complex models like Gradient Boosting, XGBoost, LightGBM, and CatBoost, consistently achieved the best results across all three dataset versions.
- Linear regression, Ridge regression, Lasso regression, and Elastic Net regression delivered similar results but fell short of the performance achieved by the Random Forest-based models and Gradient Boosting.
- The KNN model performed less effectively than other models in both dataset versions.

### Impact of Dataset Expansion

The addition of additional features to the first dataset improved prediction accuracy. However, further expansion to the third dataset version (17 features) did not yield significant improvements in model performance.

## Report

For more in-depth analysis and findings, please refer to the [report in Polish](report.pdf).
