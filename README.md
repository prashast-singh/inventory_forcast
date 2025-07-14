# Sales Forecasting Analysis Project

## 📊 Project Overview

This project implements a comprehensive sales forecasting analysis using multiple forecasting models, ensemble methods, and ABC segmentation. The analysis includes advanced data preprocessing, model selection, forecast generation, and validation for inventory management purposes.

## 🎯 Objectives

- Analyze historical sales data to understand patterns and trends
- Implement ABC segmentation for product prioritization
- Compare multiple forecasting models (ETS, Prophet, Croston, Seasonal Naive)
- Apply ensemble methods for improved accuracy
- Select optimal forecasting model for each product
- Generate 6-month forecasts for all products
- Validate forecasts using holdout testing
- Create interactive dashboard for visualization

## 📁 Project Structure

```
├── sales_data 2.csv                    # Original sales data
├── generate_forecasts_improved.py      # Main forecasting script (improved)
├── sales_forecasting_dashboard.py      # Main Streamlit dashboard
├── streamlitdash.py                    # Original Streamlit dashboard
├── forecasts_next_6_months.csv         # 6-month forecasts for all products
├── holdout_forecast_monthly.csv        # Holdout validation results
├── segment_method_comparison.csv       # Model selection and ABC segmentation
├── inventory_management_report.csv     # Enhanced inventory analysis
└── README.md                          # This file
```

## 🔧 Methodology

### 1. Data Analysis & Preprocessing

- **Data Loading**: Historical sales data with Date, Product, and Revenue columns
- **Time Series Creation**: Monthly aggregation of revenue by product
- **Advanced Preprocessing**:
  - Outlier detection and treatment
  - Missing value imputation
  - Data smoothing techniques
  - Feature engineering for seasonality

### 2. ABC Segmentation

- **Revenue-based Classification**:
  - Segment A: Top 80% of revenue (high-value products)
  - Segment B: 80-95% of revenue (medium-value products)
  - Segment C: Bottom 5% of revenue (low-value products)

### 3. Forecasting Models

- **ETS (Exponential Smoothing)**: For trend and seasonal patterns
- **Prophet**: For complex seasonality and holiday effects
- **Croston**: For intermittent demand patterns
- **Seasonal Naive**: For stable, seasonal patterns
- **Ensemble Methods**: Weighted combinations of multiple models

### 4. Model Selection & Optimization

- **Cross-validation**: Rolling window validation with optimized parameters
- **Performance Metrics**: Mean Absolute Percentage Error (MAPE), MAE, RMSE
- **Automated Selection**: Best model per product based on lowest MAPE
- **Ensemble Weighting**: Dynamic weighting based on recent performance

### 5. Forecast Generation

- **6-Month Forecasts**: Predictions for all products
- **Holdout Validation**: Testing on last 6 months of historical data
- **Performance Evaluation**: Comprehensive accuracy metrics
- **Confidence Intervals**: Uncertainty quantification

## 🚀 How to Run

### Prerequisites

```bash
pip install streamlit pandas numpy plotly statsmodels prophet scikit-learn
```

### Generate Forecasts

```bash
python generate_forecasts_improved.py
```

### Run the Dashboard

```bash
streamlit run sales_forecasting_dashboard.py
```

## 📊 Dashboard Features

### 1. Overview Section

- Key metrics and project summary
- Monthly revenue trends
- ABC segmentation distribution

### 2. Data Analysis

- Data summary statistics
- Top products by revenue
- Revenue distribution analysis

### 3. ABC Segmentation

- Pareto chart analysis
- Segment revenue distribution
- Detailed segment statistics

### 4. Forecasting Models

- Model distribution across segments
- Performance comparison (MAPE)
- Detailed model selection results

### 5. Forecasts

- Interactive product selection
- Historical vs forecast visualization
- Forecast summary statistics

### 6. Validation

- Holdout testing results
- Performance metrics
- Validation plots

### 7. Results Summary

- Key findings
- Performance metrics
- Download options for results

## 📈 Key Results

### ABC Segmentation Results

- **Segment A**: 52 products (40% of products, 80% of revenue)
- **Segment B**: 37 products (28% of products, 15% of revenue)
- **Segment C**: 41 products (32% of products, 5% of revenue)

### Model Performance (Improved)

- **Seasonal Naive**: 46 products (best for stable patterns)
- **ETS**: 35 products (best for trend/seasonal patterns)
- **Croston**: 7 products (best for intermittent demand)
- **Prophet**: 42 products (best for complex seasonality)

### Forecast Accuracy (Significantly Improved)

- **Holdout MAPE**: 37.43% (reduced from 387%)
- **Average MAE**: $6,142
- **Overall Accuracy**: Dramatically improved through ensemble methods
- **Model Reliability**: Enhanced through advanced preprocessing

## 📋 Output Files

### 1. forecasts_next_6_months.csv

Contains 6-month forecasts for all 130 products with dates and predicted revenue values.

### 2. holdout_forecast_monthly.csv

Holdout validation results showing actual vs predicted values for the last 6 months of historical data.

### 3. segment_method_comparison.csv

Complete model selection results including:

- Product names
- ABC segments
- Preferred forecasting models
- MAPE scores
- Model performance metrics

### 4. inventory_management_report.csv

Enhanced analysis including:

- Safety stock calculations
- Reorder points
- Economic order quantities
- Cost analysis

## 🎯 Business Applications

### Inventory Management

- **Safety Stock Optimization**: Based on demand variability
- **Reorder Point Planning**: Automated reorder triggers
- **Cost Optimization**: EOQ calculations for ordering

### Strategic Planning

- **Product Prioritization**: Focus on Segment A products
- **Resource Allocation**: Different strategies per segment
- **Risk Management**: Identify high-variability products

### Operational Efficiency

- **Forecast Accuracy**: Significantly improved accuracy reduces stockouts and overstock
- **Model Selection**: Automated best model per product with ensemble methods
- **Performance Monitoring**: Continuous validation and improvement

## 🔍 Technical Details

### Data Processing (Enhanced)

- **Time Series Aggregation**: Monthly revenue by product
- **Advanced Preprocessing**: Outlier detection, smoothing, feature engineering
- **Missing Value Handling**: Sophisticated imputation techniques
- **Data Quality**: Comprehensive validation and cleaning

### Model Implementation (Improved)

- **ETS**: Optimized Holt-Winters exponential smoothing
- **Prophet**: Enhanced configuration for better performance
- **Croston**: Intermittent demand forecasting with improvements
- **Seasonal Naive**: Simple seasonal pattern matching
- **Ensemble Methods**: Weighted combinations for improved accuracy

### Validation Approach (Enhanced)

- **Rolling Window**: Optimized training and testing periods
- **Performance Metrics**: MAPE, MAE, RMSE with confidence intervals
- **Cross-validation**: Multiple time periods for robustness
- **Ensemble Validation**: Comprehensive model combination testing

## 🚀 Key Improvements

### Accuracy Enhancements

- **Ensemble Methods**: Weighted model combinations
- **Advanced Preprocessing**: Better data quality
- **Optimized Parameters**: Fine-tuned model configurations

### Performance Optimizations

- **Efficient Processing**: Reduced computation time
- **Memory Management**: Optimized data handling
- **Parallel Processing**: Faster model training
- **Smart Caching**: Reduced redundant calculations

### Robustness Improvements

- **Error Handling**: Comprehensive exception management
- **Data Validation**: Quality checks throughout pipeline
- **Model Fallbacks**: Automatic model switching on failures
- **Consistency Checks**: Validation of results

## 👨‍💻 Author

Prashast Singh
Philipps University Marburg
[Date: Jul 10 2024]
