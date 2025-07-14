#!/usr/bin/env python3
"""
Complete Sales Forecasting Project
==================================

This script runs the entire sales forecasting project including:
1. Data loading and exploration
2. ABC segmentation of products
3. Multiple forecasting models (ETS, Prophet, Croston, Seasonal Naive)
4. Model selection based on cross-validation
5. Forecast generation for next 6 months
6. Holdout validation
7. Results visualization and export

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_percentage_error
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = 'sales_data 2.csv'
HORIZON = 6  # months ahead to forecast
INITIAL_TRAIN_MONTHS = 24  # rolling-CV training window
ALPHA_CROSTON = 0.1  # smoothing for Croston
SEGMENT_THRESHOLDS = {'A': 0.8, 'B': 0.95, 'C': 1.0}

def load_and_prepare(path: str) -> pd.DataFrame:
    """
    Load and prepare sales data for time series analysis.
    
    Args:
        path: Path to the CSV file
        
    Returns:
        DataFrame with monthly revenue time series (index = Date, columns = Product)
    """
    print("📊 Loading and preparing data...")
    df = pd.read_csv(path, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    
    # Aggregate to monthly level by product
    monthly = (
        df.groupby([pd.Grouper(freq='M'), 'Product'])['Revenue']
          .sum()
          .reset_index(name='Monthly_Revenue')
    )
    
    # Pivot to get products as columns
    ts = monthly.pivot(index='Date', columns='Product', values='Monthly_Revenue').fillna(0)
    ts = ts.asfreq('M')
    
    print(f"✅ Data loaded: {ts.shape[0]} months, {ts.shape[1]} products")
    return ts

def abc_segmentation(ts: pd.DataFrame) -> dict:
    """
    Perform ABC segmentation based on cumulative revenue contribution.
    
    Args:
        ts: Time series DataFrame
        
    Returns:
        Dictionary mapping product names to segment labels (A, B, C)
    """
    print("🏷️  Performing ABC segmentation...")
    total_rev = ts.sum()
    cum_share = total_rev.sort_values(ascending=False).cumsum() / total_rev.sum()
    
    segments = {}
    for prod, share in cum_share.items():
        if share <= SEGMENT_THRESHOLDS['A']:
            segments[prod] = 'A'
        elif share <= SEGMENT_THRESHOLDS['B']:
            segments[prod] = 'B'
        else:
            segments[prod] = 'C'
    
    # Print segmentation summary
    segment_counts = pd.Series(segments.values()).value_counts().sort_index()
    print("📈 ABC Segmentation Results:")
    for seg, count in segment_counts.items():
        print(f"   Segment {seg}: {count} products")
    
    return segments

def fit_prophet(series: pd.Series, horizon: int) -> pd.Series:
    """
    Fit Prophet model and generate forecasts.
    
    Args:
        series: Time series data
        horizon: Number of periods to forecast
        
    Returns:
        Forecast series
    """
    df_p = series.reset_index().rename(columns={'Date': 'ds', series.name: 'y'})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(df_p)
    future = m.make_future_dataframe(periods=horizon, freq='M')
    forecast = m.predict(future)
    return forecast.set_index('ds')['yhat'][-horizon:]

def fit_ets(series: pd.Series, horizon: int):
    """
    Fit ETS (Exponential Smoothing) model.
    
    Args:
        series: Time series data
        horizon: Number of periods to forecast
        
    Returns:
        Tuple of (fitted values, forecast)
    """
    model = ExponentialSmoothing(
        series, 
        trend='add', 
        damped_trend=True,
        seasonal='add', 
        seasonal_periods=12
    )
    fit = model.fit(optimized=True)
    return fit.fittedvalues, fit.forecast(horizon)

def croston_forecast(series: pd.Series, alpha: float, horizon: int) -> pd.Series:
    """
    Implement Croston method for intermittent demand forecasting.
    
    Args:
        series: Time series data
        alpha: Smoothing parameter
        horizon: Number of periods to forecast
        
    Returns:
        Forecast series
    """
    demand = series.values
    intervals = (demand > 0).astype(int)
    
    # Initialize
    q = demand[demand > 0].mean() if demand[demand > 0].size else 0
    p = intervals.mean() if intervals.size else 0
    
    # Update estimates
    for t in range(1, len(demand)):
        if demand[t] > 0:
            q = alpha * demand[t] + (1 - alpha) * q
            p = alpha * intervals[t] + (1 - alpha) * p
    
    # Generate forecast
    f_val = q / p if p > 0 else 0
    idx = [series.index[-1] + pd.DateOffset(months=i) for i in range(1, horizon + 1)]
    return pd.Series([f_val] * horizon, index=idx)

def seasonal_naive_forecast(series: pd.Series, horizon: int) -> pd.Series:
    """
    Generate seasonal naive forecast.
    
    Args:
        series: Time series data
        horizon: Number of periods to forecast
        
    Returns:
        Forecast series
    """
    if len(series) < 12:
        vals = np.tile(series.values, horizon)
    else:
        vals = np.tile(series[-12:].values, (horizon // 12 + 1))
    
    vals = vals[:horizon]
    idx = [series.index[-1] + pd.DateOffset(months=i) for i in range(1, horizon + 1)]
    return pd.Series(vals, index=idx)

def safe_mape(y_true, y_pred):
    """
    Calculate MAPE safely handling zero values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MAPE value or NaN if not computable
    """
    mask = y_true != 0
    if not mask.any():
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def rolling_cv(series: pd.Series, tag: str, horizon: int = HORIZON,
               initial_train: int = INITIAL_TRAIN_MONTHS) -> float:
    """
    Perform rolling cross-validation for model selection.
    
    Args:
        series: Time series data
        tag: Model type ('ETS', 'Prophet', 'Croston', 'SeasonalNaive')
        horizon: Forecast horizon
        initial_train: Initial training window size
        
    Returns:
        Average MAPE across folds
    """
    funcs = {
        'ETS': lambda s: fit_ets(s, horizon)[1],
        'Prophet': lambda s: fit_prophet(s, horizon),
        'Croston': lambda s: croston_forecast(s, ALPHA_CROSTON, horizon),
        'SeasonalNaive': lambda s: seasonal_naive_forecast(s, horizon)
    }
    
    fcast = funcs[tag]
    folds = max((len(series) - initial_train) // horizon, 0)
    errs = []
    
    for i in range(folds):
        end = initial_train + i * horizon
        train, test = series[:end], series[end:end + horizon]
        fc = fcast(train)
        errs.append(safe_mape(test.values[:len(fc)], fc.values[:len(test)]))
    
    return np.nanmean([e for e in errs if not np.isnan(e)]) if errs else np.nan

def select_preferred_models(ts: pd.DataFrame, seg: dict, n: int = 5) -> dict:
    """
    Select preferred forecasting model for each product segment.
    
    Args:
        ts: Time series DataFrame
        seg: Segmentation dictionary
        n: Number of products to evaluate per segment
        
    Returns:
        Dictionary mapping products to preferred models
    """
    print("🔍 Selecting preferred models for each segment...")
    pref = {}
    
    # Segment A: Compare ETS vs Prophet
    segment_a_products = [p for p, s in seg.items() if s == 'A'][:n]
    for sku in segment_a_products:
        ets_error = rolling_cv(ts[sku], 'ETS')
        prophet_error = rolling_cv(ts[sku], 'Prophet')
        pref[sku] = 'ETS' if ets_error <= prophet_error else 'Prophet'
        print(f"   {sku} (A): {'ETS' if ets_error <= prophet_error else 'Prophet'} "
              f"(ETS: {ets_error:.1f}%, Prophet: {prophet_error:.1f}%)")
    
    # Segment B: Compare Croston vs Seasonal Naive
    segment_b_products = [p for p, s in seg.items() if s == 'B'][:n]
    for sku in segment_b_products:
        croston_error = rolling_cv(ts[sku], 'Croston')
        naive_error = rolling_cv(ts[sku], 'SeasonalNaive')
        pref[sku] = 'Croston' if croston_error <= naive_error else 'SeasonalNaive'
        print(f"   {sku} (B): {'Croston' if croston_error <= naive_error else 'SeasonalNaive'} "
              f"(Croston: {croston_error:.1f}%, Naive: {naive_error:.1f}%)")
    
    return pref

def generate_forecasts(ts: pd.DataFrame, seg: dict, m: dict, horizon: int = HORIZON) -> pd.DataFrame:
    """
    Generate forecasts for all products using selected models.
    
    Args:
        ts: Time series DataFrame
        seg: Segmentation dictionary
        m: Model preference dictionary
        horizon: Forecast horizon
        
    Returns:
        DataFrame with forecasts for all products
    """
    print("🚀 Generating forecasts for all products...")
    out = []
    
    for prod, seg_tag in seg.items():
        s = ts[prod]
        tag = m.get(prod, 
                   'ETS' if seg_tag == 'A' else 
                   'Croston' if seg_tag == 'B' else 'SeasonalNaive')
        
        if tag == 'Prophet':
            fc = fit_prophet(s, horizon)
        elif tag == 'ETS':
            fc = fit_ets(s, horizon)[1]
        elif tag == 'Croston':
            fc = croston_forecast(s, ALPHA_CROSTON, horizon)
        else:  # SeasonalNaive
            fc = seasonal_naive_forecast(s, horizon)
        
        fc.name = prod
        out.append(fc)
    
    return pd.concat(out, axis=1)

def create_visualizations(ts: pd.DataFrame, forecasts: pd.DataFrame, segments: dict, 
                         preferred_models: dict, target_sku: str = None):
    """
    Create comprehensive visualizations for the forecasting project.
    
    Args:
        ts: Historical time series data
        forecasts: Forecast data
        segments: ABC segmentation
        preferred_models: Model preferences
        target_sku: Specific product to visualize (if None, use first A segment product)
    """
    print("📊 Creating visualizations...")
    
    # Set target SKU if not provided
    if target_sku is None:
        target_sku = [p for p, s in segments.items() if s == 'A'][0]
    
    if target_sku not in ts.columns:
        print(f"⚠️  Target SKU '{target_sku}' not found. Using first available product.")
        target_sku = ts.columns[0]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Revenue by Product Category', 'ABC Segmentation Distribution',
                       f'Forecast vs Actual: {target_sku}', 'Monthly Revenue Trend'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. Product Category Revenue
    category_revenue = ts.sum().groupby(lambda x: x.split(',')[0] if ',' in x else x).sum()
    fig.add_trace(
        go.Bar(x=category_revenue.index, y=category_revenue.values, name='Category Revenue'),
        row=1, col=1
    )
    
    # 2. ABC Segmentation Distribution
    segment_counts = pd.Series(segments.values()).value_counts()
    fig.add_trace(
        go.Pie(labels=segment_counts.index, values=segment_counts.values, name='Segments'),
        row=1, col=2
    )
    
    # 3. Forecast vs Actual for target SKU
    actual = ts[target_sku].dropna()
    forecast = forecasts[target_sku]
    
    fig.add_trace(
        go.Scatter(x=actual.index, y=actual.values, mode='lines', name='Actual', line=dict(color='blue')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=forecast.index, y=forecast.values, mode='lines+markers', 
                  name='Forecast', line=dict(color='red')),
        row=2, col=1
    )
    
    # 4. Monthly Revenue Trend
    monthly_total = ts.sum(axis=1)
    fig.add_trace(
        go.Scatter(x=monthly_total.index, y=monthly_total.values, mode='lines', 
                  name='Total Revenue', line=dict(color='green')),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="Sales Forecasting Project - Comprehensive Dashboard",
        showlegend=True
    )
    
    # Save the plot
    fig.write_html("forecasting_dashboard.html")
    print("✅ Dashboard saved as 'forecasting_dashboard.html'")
    
    return fig

def main():
    """
    Main function to run the complete forecasting project.
    """
    print("🎯 Starting Complete Sales Forecasting Project")
    print("=" * 50)
    
    # 1. Load and prepare data
    ts = load_and_prepare(DATA_PATH)
    
    # 2. ABC segmentation
    segments = abc_segmentation(ts)
    
    # 3. Select preferred models
    preferred_models = select_preferred_models(ts, segments)
    
    # 4. Generate forecasts
    forecasts = generate_forecasts(ts, segments, preferred_models)
    
    # 5. Create segment comparison CSV
    segment_df = pd.DataFrame({
        'Product': list(preferred_models.keys()),
        'Preferred': list(preferred_models.values())
    })
    segment_df['Segment'] = segment_df['Product'].map(segments)
    segment_df.to_csv('segment_method_comparison.csv', index=False)
    print("📄 segment_method_comparison.csv saved")
    
    # 6. Save forecasts
    forecasts.to_csv(f'forecasts_next_{HORIZON}_months.csv')
    print(f"✅ forecasts_next_{HORIZON}_months.csv saved")
    
    # 7. Generate holdout forecasts
    print("🔬 Generating holdout forecasts for validation...")
    ho_rows = []
    for prod in ts.columns:
        series = ts[prod]
        train_hist = series[:-HORIZON]  # training up to Jan-2016
        
        tag = preferred_models.get(prod, 
                                 'ETS' if segments[prod]=='A' else 
                                 'Croston' if segments[prod]=='B' else 'SeasonalNaive')
        
        if tag == 'Prophet':
            fc = fit_prophet(train_hist, HORIZON)
        elif tag == 'ETS':
            fc = fit_ets(train_hist, HORIZON)[1]
        elif tag == 'Croston':
            fc = croston_forecast(train_hist, ALPHA_CROSTON, HORIZON)
        else:  # SeasonalNaive
            fc = seasonal_naive_forecast(train_hist, HORIZON)
        
        for dt, val in fc.items():
            ho_rows.append({
                'Product': prod,
                'Date': dt.strftime('%Y-%m-%d'),
                'Forecast': val
            })
    
    pd.DataFrame(ho_rows).to_csv('holdout_forecast_monthly.csv', index=False)
    print("📄 holdout_forecast_monthly.csv saved")
    
    # 8. Create visualizations
    create_visualizations(ts, forecasts, segments, preferred_models)
    
    # 9. Print summary statistics
    print("\n📊 Project Summary:")
    print(f"   Total Products: {len(ts.columns)}")
    print(f"   Time Period: {ts.index.min().strftime('%Y-%m')} to {ts.index.max().strftime('%Y-%m')}")
    print(f"   Forecast Horizon: {HORIZON} months")
    print(f"   Total Revenue: ${ts.sum().sum():,.0f}")
    
    # Model usage summary
    model_usage = pd.Series(preferred_models.values()).value_counts()
    print("\n🔧 Model Usage Summary:")
    for model, count in model_usage.items():
        print(f"   {model}: {count} products")
    
    print("\n🎉 Forecasting project completed successfully!")
    print("📁 Generated files:")
    print("   - segment_method_comparison.csv")
    print("   - forecasts_next_6_months.csv")
    print("   - holdout_forecast_monthly.csv")
    print("   - forecasting_dashboard.html")
    print("\n🚀 To run the Streamlit dashboard:")
    print("   streamlit run streamlitdash.py")

if __name__ == '__main__':
    main() 