#!/usr/bin/env python3
"""
Improved Sales Forecasting Data Generation Script
================================================

This script generates all required CSV files with comprehensive accuracy improvements:
1. segment_method_comparison.csv - ABC segmentation and enhanced model selection
2. forecasts_next_6_months.csv - 6-month forecasts for all products
3. holdout_forecast_monthly.csv - Holdout validation results
4. inventory_management_report.csv - Enhanced inventory analysis

Key Improvements:
- Ensemble methods for better accuracy
- Advanced data preprocessing (outlier removal, smoothing)
- Enhanced Prophet configuration
- Better model selection criteria
- MAE instead of MAPE for evaluation
- Product-specific optimization
- Weighted ensemble forecasting

Author: [Your Name]
Date: 2024
"""

import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare the sales data with enhanced preprocessing."""
    print("📊 Loading and preprocessing sales data...")
    
    # Load the original sales data
    sales_df = pd.read_csv('sales_data 2.csv', parse_dates=['Date'])
    
    # Aggregate by product and month
    sales_df['YearMonth'] = sales_df['Date'].dt.to_period('M')
    ts_data = sales_df.groupby(['Product', 'YearMonth'])['Revenue'].sum().reset_index()
    ts_data['YearMonth'] = ts_data['YearMonth'].astype(str)
    ts_data['Date'] = pd.to_datetime(ts_data['YearMonth'])
    
    # Create time series matrix
    ts_matrix = ts_data.pivot(index='Date', columns='Product', values='Revenue').fillna(0)
    
    print(f"✅ Loaded data for {len(ts_matrix.columns)} products")
    min_idx = ts_matrix.index.min()
    max_idx = ts_matrix.index.max()
    min_str = min_idx.strftime('%Y-%m') if isinstance(min_idx, pd.Timestamp) else str(min_idx)
    max_str = max_idx.strftime('%Y-%m') if isinstance(max_idx, pd.Timestamp) else str(max_idx)
    print(f"📅 Time period: {min_str} to {max_str}")
    
    return ts_matrix, sales_df

def preprocess_time_series(ts_matrix):
    """Apply advanced preprocessing to time series data."""
    print("🔧 Applying advanced data preprocessing...")
    
    def preprocess_series(series):
        """Apply preprocessing to a single time series."""
        # Remove outliers using IQR method
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap outliers
        series_clean = series.copy()
        series_clean[series_clean < lower_bound] = lower_bound
        series_clean[series_clean > upper_bound] = upper_bound
        
        # Apply smoothing for high variability series
        cv = series_clean.std() / series_clean.mean() if series_clean.mean() > 0 else 0
        if cv > 2:  # High coefficient of variation
            series_clean = series_clean.rolling(window=3, center=True, min_periods=1).mean()
        
        return series_clean
    
    # Apply preprocessing to each product
    ts_matrix_clean = ts_matrix.copy()
    for product in ts_matrix.columns:
        ts_matrix_clean[product] = preprocess_series(ts_matrix[product])
    
    print("   ✅ Data preprocessing completed")
    return ts_matrix_clean

def perform_abc_segmentation(ts_matrix):
    """Perform ABC segmentation based on total revenue."""
    print("🏷️ Performing ABC segmentation...")
    
    total_revenue = ts_matrix.sum()
    total_revenue_sorted = total_revenue.sort_values(ascending=False)
    
    # Calculate cumulative percentages
    cumulative_revenue = total_revenue_sorted.cumsum()
    total_sum = total_revenue_sorted.sum()
    cumulative_percentage = cumulative_revenue / total_sum * 100
    
    # Assign segments
    segments = {}
    for product in total_revenue_sorted.index:
        if cumulative_percentage[product] <= 80:
            segments[product] = 'A'
        elif cumulative_percentage[product] <= 95:
            segments[product] = 'B'
        else:
            segments[product] = 'C'
    
    # Create segmentation summary
    segment_counts = pd.Series(segments.values()).value_counts()
    print(f"   Segment A: {segment_counts.get('A', 0)} products")
    print(f"   Segment B: {segment_counts.get('B', 0)} products")
    print(f"   Segment C: {segment_counts.get('C', 0)} products")
    
    return segments, total_revenue_sorted

def check_data_quality(ts_series):
    """Enhanced data quality check."""
    if len(ts_series) < 18:  # Minimum 18 months for rolling CV
        return False, f"Insufficient data ({len(ts_series)} months < 18)"
    
    if ts_series.std() == 0:
        return False, "No variance in data"
    
    if ts_series.sum() == 0:
        return False, "All zero values"
    
    # Check for too many zeros (intermittent demand)
    zero_ratio = (ts_series == 0).sum() / len(ts_series)
    if zero_ratio > 0.9:  # Increased threshold
        return False, f"Too many zeros ({zero_ratio:.1%})"
    
    return True, "OK"

def croston_forecast(train, horizon):
    """Improved Croston implementation for intermittent demand."""
    non_zero = train[train > 0]
    if len(non_zero) < 3:
        return np.full(horizon, train.mean())
    
    # Simple exponential smoothing for non-zero values
    alpha = 0.3
    if len(non_zero) > 1:
        forecast_value = non_zero.iloc[-1] * alpha + (1 - alpha) * non_zero.mean()
    else:
        forecast_value = non_zero.iloc[0]
    
    return np.full(horizon, forecast_value)

def enhanced_rolling_cv(ts_series, model_type, window=6, horizon=6):
    """Enhanced rolling cross-validation with better error handling."""
    if len(ts_series) < window + horizon:
        return float('inf')
    
    # Check data quality first
    is_valid, reason = check_data_quality(ts_series)
    if not is_valid:
        return float('inf')
    
    errors = []
    for i in range(window, len(ts_series) - horizon + 1):
        train = ts_series.iloc[:i]
        test = ts_series.iloc[i:i+horizon]
        
        try:
            if model_type == 'ETS_Enhanced':
                # Enhanced ETS with better parameter selection
                if len(train) >= 24:
                    model = ExponentialSmoothing(train, seasonal_periods=12, seasonal='add')
                else:
                    model = ExponentialSmoothing(train, seasonal=None)
                fitted_model = model.fit()
                forecast = fitted_model.forecast(horizon)
                
            elif model_type == 'Prophet_Enhanced':
                # Enhanced Prophet with optimized configuration
                if len(train) >= 18:
                    df_prophet = pd.DataFrame({'ds': train.index, 'y': train.values})
                    model = Prophet(
                        yearly_seasonality=False,
                        weekly_seasonality=False,
                        daily_seasonality=False,
                        seasonality_mode='additive',
                        changepoint_prior_scale=0.01,  # Less flexible for stability
                        seasonality_prior_scale=1.0    # More seasonal
                    )
                    
                    # Suppress Prophet output
                    import sys
                    from io import StringIO
                    old_stdout = sys.stdout
                    sys.stdout = StringIO()
                    
                    try:
                        model.fit(df_prophet)
                        future = model.make_future_dataframe(periods=horizon, freq='M')
                        forecast = model.predict(future)['yhat'].iloc[-horizon:].values
                    finally:
                        sys.stdout = old_stdout
                else:
                    continue
                    
            elif model_type == 'Ensemble':
                # Weighted ensemble of multiple models
                forecasts = []
                weights = []
                
                # ETS forecast
                try:
                    if len(train) >= 12:
                        model = ExponentialSmoothing(train, seasonal=None)
                        fitted_model = model.fit()
                        ets_forecast = fitted_model.forecast(horizon)
                        forecasts.append(ets_forecast)
                        weights.append(0.4)  # Higher weight for ETS
                except:
                    pass
                
                # Moving average forecast
                if len(train) >= 6:
                    ma_forecast = np.full(horizon, train.tail(6).mean())
                    forecasts.append(ma_forecast)
                    weights.append(0.3)
                
                # Simple exponential smoothing
                if len(train) >= 3:
                    ses_forecast = np.full(horizon, train.iloc[-1] * 0.3 + train.mean() * 0.7)
                    forecasts.append(ses_forecast)
                    weights.append(0.3)
                
                # Use weighted average of available forecasts
                if forecasts:
                    weights = np.array(weights) / sum(weights)  # Normalize weights
                    forecast = np.average(forecasts, axis=0, weights=weights)
                else:
                    forecast = np.full(horizon, train.mean())
                    
            elif model_type == 'Croston':
                forecast = croston_forecast(train, horizon)
                
            elif model_type == 'SeasonalNaive':
                if len(train) >= 12:
                    forecast = np.tile(train.iloc[-12:].values, (horizon // 12 + 1))[:horizon]
                else:
                    forecast = np.full(horizon, train.mean())
            else:
                continue
                
            # Use MAE instead of MAPE for better handling of small values
            if len(test) > 0:
                mae = mean_absolute_error(test, forecast)
                errors.append(mae)
                
        except Exception as e:
            continue
    
    if not errors:
        return float('inf')
    
    return np.mean(errors)

def select_enhanced_models(ts_matrix, segments):
    """Enhanced model selection with better criteria."""
    print("🔍 Selecting enhanced models for each product...")
    preferred_models = {}
    model_performance = []
    fallback_count = 0
    prophet_count = 0
    ensemble_count = 0
    
    for i, (product, segment) in enumerate(segments.items()):
        print(f"   Processing {i+1}/{len(segments)}: {product} (Segment {segment})")
        ts_series = ts_matrix[product]
        print(f"      Series length: {len(ts_series)}")
        
        # Check data quality first
        is_valid, reason = check_data_quality(ts_series)
        if not is_valid:
            print(f"      [SKIP] Data quality issue: {reason}")
            best_model = 'Ensemble'
            best_mae = enhanced_rolling_cv(ts_series, 'Ensemble')
        else:
            # Define models based on segment and data characteristics
            if segment == 'A' and len(ts_series) >= 24:
                models_to_compare = ['ETS_Enhanced', 'Prophet_Enhanced', 'Ensemble']
            elif segment == 'A':
                models_to_compare = ['ETS_Enhanced', 'Ensemble']
            elif segment == 'B' and len(ts_series) >= 18:
                models_to_compare = ['ETS_Enhanced', 'Prophet_Enhanced', 'Ensemble']
            elif segment == 'B':
                models_to_compare = ['ETS_Enhanced', 'Ensemble']
            else:  # Segment C
                models_to_compare = ['Ensemble', 'SeasonalNaive']
            
            best_model = None
            best_mae = float('inf')
            
            for model in models_to_compare:
                mae = enhanced_rolling_cv(ts_series, model)
                print(f"      {model} MAE: ${mae:.2f}")
                if mae < best_mae:
                    best_mae = mae
                    best_model = model
            
            # Fallback for extremely high MAE
            if best_mae > 50000 or best_model is None:  # High threshold for MAE
                print(f"   [FALLBACK] Using Ensemble for {product} (best_mae=${best_mae:.2f})")
                fallback_count += 1
                best_model = 'Ensemble'
                best_mae = enhanced_rolling_cv(ts_series, 'Ensemble')
        
        if best_model == 'Prophet_Enhanced':
            prophet_count += 1
        elif best_model == 'Ensemble':
            ensemble_count += 1
            
        preferred_models[product] = best_model
        model_performance.append({
            'Product': product,
            'Segment': segment,
            'Preferred_Model': best_model,
            'MAE': best_mae
        })
    
    print(f"\n[SUMMARY] Enhanced Model Selection Results:")
    print(f"   Fallbacks to Ensemble: {fallback_count} out of {len(segments)} products")
    print(f"   Prophet models used: {prophet_count} products")
    print(f"   Ensemble models used: {ensemble_count} products")
    
    return pd.DataFrame(model_performance), preferred_models

def generate_enhanced_forecasts(ts_matrix, preferred_models):
    """Generate 6-month forecasts using enhanced models."""
    print("🔮 Generating enhanced 6-month forecasts...")
    
    # Get the last date from the data
    last_date = ts_matrix.index[-1]
    
    # Create future dates (next 6 months)
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                periods=6, freq='M')
    
    forecasts = {'Date': future_dates}
    
    for product in ts_matrix.columns:
        ts_series = ts_matrix[product]
        model_type = preferred_models[product]
        
        try:
            if model_type == 'ETS_Enhanced':
                if len(ts_series) >= 24:
                    model = ExponentialSmoothing(ts_series, seasonal_periods=12, seasonal='add')
                else:
                    model = ExponentialSmoothing(ts_series, seasonal=None)
                fitted_model = model.fit()
                forecast = fitted_model.forecast(6)
                
            elif model_type == 'Prophet_Enhanced':
                df_prophet = pd.DataFrame({'ds': ts_series.index, 'y': ts_series.values})
                model = Prophet(
                    yearly_seasonality=False,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    seasonality_mode='additive',
                    changepoint_prior_scale=0.01,
                    seasonality_prior_scale=1.0
                )
                
                # Suppress Prophet output
                import sys
                from io import StringIO
                old_stdout = sys.stdout
                sys.stdout = StringIO()
                
                try:
                    model.fit(df_prophet)
                    future = model.make_future_dataframe(periods=6, freq='M')
                    forecast = model.predict(future)['yhat'].iloc[-6:].values
                finally:
                    sys.stdout = old_stdout
                
            elif model_type == 'Ensemble':
                # Weighted ensemble forecasting
                forecasts_list = []
                weights = []
                
                # ETS component
                try:
                    if len(ts_series) >= 12:
                        model = ExponentialSmoothing(ts_series, seasonal=None)
                        fitted_model = model.fit()
                        ets_forecast = fitted_model.forecast(6)
                        forecasts_list.append(ets_forecast)
                        weights.append(0.4)
                except:
                    pass
                
                # Moving average component
                if len(ts_series) >= 6:
                    ma_forecast = np.full(6, ts_series.tail(6).mean())
                    forecasts_list.append(ma_forecast)
                    weights.append(0.3)
                
                # Simple exponential smoothing component
                if len(ts_series) >= 3:
                    ses_forecast = np.full(6, ts_series.iloc[-1] * 0.3 + ts_series.mean() * 0.7)
                    forecasts_list.append(ses_forecast)
                    weights.append(0.3)
                
                # Use weighted average
                if forecasts_list:
                    weights = np.array(weights) / sum(weights)
                    forecast = np.average(forecasts_list, axis=0, weights=weights)
                else:
                    forecast = np.full(6, ts_series.mean())
                
            elif model_type == 'Croston':
                forecast = croston_forecast(ts_series, 6)
                
            elif model_type == 'SeasonalNaive':
                if len(ts_series) >= 12:
                    forecast = np.tile(ts_series.iloc[-12:].values, 1)[:6]
                else:
                    forecast = np.full(6, ts_series.mean())
            else:
                forecast = np.full(6, ts_series.mean())
            
            forecasts[product] = forecast.tolist()
            
        except Exception as e:
            print(f"[ERROR] Forecast generation failed for {product}: {str(e)[:100]}")
            # Fallback to simple mean
            forecasts[product] = np.full(6, ts_series.mean()).tolist()
    
    # Build forecasts DataFrame efficiently
    forecast_matrix = []
    for i, date in enumerate(future_dates):
        row = {'Date': date}
        for product in ts_matrix.columns:
            row[product] = forecasts[product][i] if product in forecasts else None
        row['Date'] = date
        forecast_matrix.append(row)
    df_forecasts = pd.DataFrame(forecast_matrix)
    return df_forecasts

def generate_enhanced_holdout_validation(ts_matrix, preferred_models):
    """Generate enhanced holdout validation results."""
    print("✅ Generating enhanced holdout validation...")
    
    # Use last 6 months as holdout period
    holdout_period = 6
    train_data = ts_matrix.iloc[:-holdout_period]
    test_data = ts_matrix.iloc[-holdout_period:]
    
    validation_results = []
    
    for product in ts_matrix.columns:
        ts_series = train_data[product]
        actual_values = test_data[product]
        model_type = preferred_models[product]
        
        try:
            if model_type == 'ETS_Enhanced':
                if len(ts_series) >= 24:
                    model = ExponentialSmoothing(ts_series, seasonal_periods=12, seasonal='add')
                else:
                    model = ExponentialSmoothing(ts_series, seasonal=None)
                fitted_model = model.fit()
                forecast = fitted_model.forecast(holdout_period)
                
            elif model_type == 'Prophet_Enhanced':
                df_prophet = pd.DataFrame({'ds': ts_series.index, 'y': ts_series.values})
                model = Prophet(
                    yearly_seasonality=False,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    seasonality_mode='additive',
                    changepoint_prior_scale=0.01,
                    seasonality_prior_scale=1.0
                )
                
                # Suppress Prophet output
                import sys
                from io import StringIO
                old_stdout = sys.stdout
                sys.stdout = StringIO()
                
                try:
                    model.fit(df_prophet)
                    future = model.make_future_dataframe(periods=holdout_period, freq='M')
                    forecast = model.predict(future)['yhat'].iloc[-holdout_period:].values
                finally:
                    sys.stdout = old_stdout
                
            elif model_type == 'Ensemble':
                # Weighted ensemble for holdout validation
                forecasts_list = []
                weights = []
                
                # ETS component
                try:
                    if len(ts_series) >= 12:
                        model = ExponentialSmoothing(ts_series, seasonal=None)
                        fitted_model = model.fit()
                        ets_forecast = fitted_model.forecast(holdout_period)
                        forecasts_list.append(ets_forecast)
                        weights.append(0.4)
                except:
                    pass
                
                # Moving average component
                if len(ts_series) >= 6:
                    ma_forecast = np.full(holdout_period, ts_series.tail(6).mean())
                    forecasts_list.append(ma_forecast)
                    weights.append(0.3)
                
                # Simple exponential smoothing component
                if len(ts_series) >= 3:
                    ses_forecast = np.full(holdout_period, ts_series.iloc[-1] * 0.3 + ts_series.mean() * 0.7)
                    forecasts_list.append(ses_forecast)
                    weights.append(0.3)
                
                # Use weighted average
                if forecasts_list:
                    weights = np.array(weights) / sum(weights)
                    forecast = np.average(forecasts_list, axis=0, weights=weights)
                else:
                    forecast = np.full(holdout_period, ts_series.mean())
                
            elif model_type == 'Croston':
                forecast = croston_forecast(ts_series, holdout_period)
                
            elif model_type == 'SeasonalNaive':
                if len(ts_series) >= 12:
                    forecast = np.tile(ts_series.iloc[-12:].values, 1)[:holdout_period]
                else:
                    forecast = np.full(holdout_period, ts_series.mean())
            else:
                forecast = np.full(holdout_period, ts_series.mean())
            
            # Calculate metrics
            for i, (actual, pred) in enumerate(zip(actual_values, forecast)):
                error = actual - pred
                mae = abs(error)
                mape = abs(error / actual) * 100 if actual > 0 else 0
                mape = min(mape, 1000)  # Cap MAPE at 1000%
                
                validation_results.append({
                    'Date': test_data.index[i],
                    'Product': product,
                    'Actual': actual,
                    'Forecast': pred,
                    'Error': error,
                    'MAE': mae,
                    'MAPE': mape
                })
                
        except Exception as e:
            print(f"[ERROR] Holdout validation failed for {product}: {str(e)[:100]}")
            # Fallback
            for i, actual in enumerate(actual_values):
                pred = ts_series.mean()
                error = actual - pred
                mae = abs(error)
                mape = abs(error / actual) * 100 if actual > 0 else 0
                mape = min(mape, 1000)
                
                validation_results.append({
                    'Date': test_data.index[i],
                    'Product': product,
                    'Actual': actual,
                    'Forecast': pred,
                    'Error': error,
                    'MAE': mae,
                    'MAPE': mape
                })
    
    return pd.DataFrame(validation_results)

def generate_enhanced_inventory_report(ts_matrix, segments, preferred_models, sales_df):
    """Generate enhanced inventory management report."""
    print("📊 Generating enhanced inventory management report...")
    
    inventory_data = []
    
    for product in ts_matrix.columns:
        ts_series = ts_matrix[product]
        segment = segments[product]
        model = preferred_models[product]
        
        # Get actual unit cost and price from sales data
        product_data = sales_df[sales_df['Product'] == product]
        if len(product_data) > 0:
            unit_cost = product_data['Unit_Cost'].iloc[0]
            unit_price = product_data['Unit_Price'].iloc[0]
            unit_profit = unit_price - unit_cost
        else:
            unit_cost = 100  # fallback
            unit_price = 150  # fallback
            unit_profit = 50  # fallback
        
        # Calculate demand statistics
        mean_daily_demand = ts_series.mean()
        demand_std = ts_series.std()
        demand_cv = demand_std / mean_daily_demand if mean_daily_demand > 0 else 0
        
        # Determine demand consistency
        if demand_cv < 0.5:
            consistency = 'High'
        elif demand_cv < 1.0:
            consistency = 'Medium'
        else:
            consistency = 'Low'
        
        # Calculate safety stock (simplified)
        z_score = 1.96  # 95% service level
        lead_time = 30  # days
        safety_stock = z_score * np.sqrt(lead_time) * demand_std / np.sqrt(30)
        
        # Calculate reorder point
        reorder_point = mean_daily_demand * lead_time + safety_stock
        
        # Calculate EOQ using actual unit cost
        annual_demand = mean_daily_demand * 365
        ordering_cost = 50  # assumed ordering cost
        holding_cost_rate = 0.2  # 20% of unit cost
        eoq = np.sqrt((2 * annual_demand * ordering_cost) / (holding_cost_rate * unit_cost))
        
        # Calculate total cost using actual unit cost
        total_cost = (annual_demand / eoq) * ordering_cost + (eoq / 2) * holding_cost_rate * unit_cost
        cost_per_unit = total_cost / annual_demand if annual_demand > 0 else 0
        
        # Calculate profit potential
        annual_revenue = annual_demand * unit_price
        annual_profit = annual_demand * unit_profit
        
        inventory_data.append({
            'Product': product,
            'Segment': segment,
            'Preferred_Model': model,
            'Unit_Cost': unit_cost,
            'Unit_Price': unit_price,
            'Unit_Profit': unit_profit,
            'Mean_Daily_Demand': mean_daily_demand,
            'Demand_CV': demand_cv,
            'Demand_Consistency': consistency,
            'Safety_Stock': safety_stock,
            'Reorder_Point': reorder_point,
            'EOQ': eoq,
            'Total_Cost': total_cost,
            'Cost_Per_Unit': cost_per_unit,
            'Annual_Revenue': annual_revenue,
            'Annual_Profit': annual_profit
        })
    
    return pd.DataFrame(inventory_data)

def main():
    """Main function to generate all CSV files with enhanced accuracy."""
    print("🚀 Starting Improved Sales Forecasting Data Generation")
    print("=" * 70)
    
    # 1. Load and prepare data
    ts_matrix, sales_df = load_and_prepare_data()
    
    # 2. Apply advanced preprocessing
    ts_matrix_clean = preprocess_time_series(ts_matrix)
    
    # 3. Perform ABC segmentation
    segments, total_revenue = perform_abc_segmentation(ts_matrix_clean)
    
    # 4. Select enhanced models
    model_comparison_df, preferred_models = select_enhanced_models(ts_matrix_clean, segments)
    
    # 5. Generate enhanced forecasts
    forecasts_df = generate_enhanced_forecasts(ts_matrix_clean, preferred_models)
    
    # 6. Generate enhanced holdout validation
    holdout_df = generate_enhanced_holdout_validation(ts_matrix_clean, preferred_models)
    
    # 7. Generate enhanced inventory report
    inventory_df = generate_enhanced_inventory_report(ts_matrix_clean, segments, preferred_models, sales_df)
    
    # 8. Save all files
    print("\n💾 Saving output files...")
    
    # Save segment method comparison
    model_comparison_df.to_csv('segment_method_comparison.csv', index=False)
    print("   ✅ segment_method_comparison.csv saved")
    
    # Save forecasts
    forecasts_df.to_csv('forecasts_next_6_months.csv', index=False)
    print("   ✅ forecasts_next_6_months.csv saved")
    
    # Save holdout validation
    holdout_df.to_csv('holdout_forecast_monthly.csv', index=False)
    print("   ✅ holdout_forecast_monthly.csv saved")
    
    # Save inventory report
    inventory_df.to_csv('inventory_management_report.csv', index=False)
    print("   ✅ inventory_management_report.csv saved")
    
    # 9. Print summary
    print("\n📊 Generation Summary:")
    print(f"   📦 Total Products: {len(ts_matrix_clean.columns)}")
    print(f"   🏷️  ABC Segments: {pd.Series(segments.values()).value_counts().to_dict()}")
    print(f"   📈 Models Used: {pd.Series(preferred_models.values()).value_counts().to_dict()}")
    print(f"   🔮 Forecast Period: 6 months")
    print(f"   ✅ Validation: Enhanced holdout testing completed")
    
    # Model performance summary
    avg_mae = model_comparison_df['MAE'].mean()
    print(f"   📊 Average MAE: ${avg_mae:.2f}")
    
    # Segment performance
    segment_performance = model_comparison_df.groupby('Segment')['MAE'].agg(['mean', 'count'])
    print(f"\n📈 Segment Performance:")
    for segment in ['A', 'B', 'C']:
        if segment in segment_performance.index:
            mean_mae = segment_performance.loc[segment, 'mean']
            count = segment_performance.loc[segment, 'count']
            print(f"   Segment {segment}: ${mean_mae:.2f} MAE ({count} products)")
    
    # Holdout validation summary
    overall_mae = holdout_df['MAE'].mean()
    overall_mape = holdout_df['MAPE'].mean()
    print(f"\n✅ Holdout Validation Results:")
    print(f"   Overall MAE: ${overall_mae:.2f}")
    print(f"   Overall MAPE: {overall_mape:.2f}%")
    
    print("\n🎉 All CSV files generated successfully with enhanced accuracy!")
    print("\n📁 Generated Files:")
    print("   - segment_method_comparison.csv")
    print("   - forecasts_next_6_months.csv")
    print("   - holdout_forecast_monthly.csv")
    print("   - inventory_management_report.csv")

if __name__ == "__main__":
    main() 