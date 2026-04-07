# Regression Path

This folder keeps the regression material in one smaller learning path instead of spreading the same ideas across duplicate notebooks.

## Core notebooks

1. `01_simple_linear_regression.ipynb`
   - one feature, one target
   - train/test split
   - `LinearRegression`
   - R2, adjusted R2, MSE, MAE
   - OLS comparison

2. `02_multiple_linear_regression.ipynb`
   - multiple input features
   - assumption checks
   - scaling
   - OLS summary
   - cross-validation

3. `03_polynomial_regression.ipynb`
   - nonlinear fit with `PolynomialFeatures`
   - train/test workflow
   - pipeline thinking

4. `04_regularization_data_prep.ipynb`
   - dataset cleaning
   - feature preparation
   - EDA before regularized regression

5. `05_regularized_regression_training.ipynb`
   - standardization
   - linear regression baseline
   - Ridge, Lasso, ElasticNet
   - `RidgeCV`, `LassoCV`, `ElasticNetCV`

## Datasets used here

- `height-weight.csv`
- `economic_index.csv`
- `Algerian_forest_fires_dataset_UPDATE.csv`
- `Algerian_forest_fires_cleaned_dataset.csv`

## Reference notes

The `reference_notes/` folder keeps only the PDFs that still add something unique to the curated set:

- OLS basics
- error metrics
- overfitting and underfitting
- polynomial regression
- Ridge/Lasso/ElasticNet
- cross-validation
