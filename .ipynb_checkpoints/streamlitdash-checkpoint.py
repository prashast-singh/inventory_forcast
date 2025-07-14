import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Product Forecast Dashboard", layout="wide")
st.title("📈 Product Forecast Dashboard")

# ---------- paths ----------
HIS_CSV = "sales_data 2.csv"            # raw transaction file
FC_CSV  = "forecasts_next_6_months.csv"
SEG_CSV = "segment_method_comparison.csv"   # optional

HO_CSV = "holdout_forecast_monthly.csv"

@st.cache_data
def load_holdout(path: str) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, parse_dates=["Date"])   # Product · Date · Forecast
    except FileNotFoundError:
        return None

holdout = load_holdout(HO_CSV)

# ---------- 1 · Load data ----------
@st.cache_data
def load_hist(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    monthly = (df.groupby([pd.Grouper(freq="M"), "Product"])["Revenue"]
                 .sum()
                 .reset_index(name="Actual"))
    return (monthly
            .pivot(index="Date", columns="Product", values="Actual")
            .asfreq("M")
            .fillna(0))

@st.cache_data
def load_forecasts(path: str) -> pd.DataFrame:
    df = (pd.read_csv(path, parse_dates=["Unnamed: 0"])
            .rename(columns={"Unnamed: 0": "Date"})
            .set_index("Date"))
    return df

@st.cache_data
def load_segments(path: str) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)        # Product · Preferred · Segment
    except FileNotFoundError:
        return None

hist       = load_hist(HIS_CSV)
forecasts  = load_forecasts(FC_CSV)
segments   = load_segments(SEG_CSV)

# ---------- 2 · Sidebar filters ----------
products = forecasts.columns.tolist()

if segments is not None:
    seg_filter = st.sidebar.multiselect("Segment", ["A", "B", "C"], default=["A", "B", "C"])
    allowed = segments[segments["Segment"].isin(seg_filter)]["Product"]
    default_selection = allowed.head(10).tolist() or products[:10]
else:
    default_selection = products[:10]

selected_products = st.sidebar.multiselect("Products", products, default=default_selection)

# ---------- 4 · Forecast vs Actual (+ hold-out) ------------------------
st.subheader("Forecast vs Actual")
sku = st.selectbox("Choose a product", selected_products)

actual    = hist[sku].dropna()
prod_fc   = forecasts[sku]

df_actual  = actual.reset_index().rename(columns={sku: "Value"})
df_actual["Series"] = "Actual"

df_fore    = prod_fc.reset_index().rename(columns={sku: "Value"})
df_fore["Series"]  = "Forecast"

frames = [df_actual, df_fore]

# add hold-out rows if file present
if holdout is not None and sku in holdout["Product"].values:
    df_ho = (holdout[holdout["Product"] == sku]
             .rename(columns={"Forecast": "Value"})
             .loc[:, ["Date", "Value"]]
             .assign(Series="Hold-out Forecast"))
    frames.append(df_ho)

plot_df = pd.concat(frames, ignore_index=True)

fig = px.line(plot_df,
              x="Date", y="Value", color="Series", markers=True,
              title=f"{sku}: Actual • Forecast • Hold-out Forecast",
              category_orders={"Series": ["Actual", "Forecast", "Hold-out Forecast"]})
fig.update_layout(xaxis_title="Date",
                  yaxis_title="Revenue",
                  yaxis_tickformat=",")

# vertical guide at start of production forecast
fig.add_vline(x=prod_fc.index.min(), line_dash="dot", line_color="gray")

st.plotly_chart(fig, use_container_width=True)