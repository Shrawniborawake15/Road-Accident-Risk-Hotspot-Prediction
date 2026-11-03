"""
Road Accident Risk Hotspot Prediction - PBL Prototype
Sanjivani University, Dept. of AI & Data Science

Run:  pip install -r requirements.txt
      streamlit run app.py

Models: Linear Regression (grouped accident counts) and
        Logistic Regression (high-risk vs lower-risk label).
Demo data is SYNTHETIC. Upload a real row-level CSV for real results.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score)
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Accident Hotspot Prediction", page_icon="🚧", layout="wide")

FEATURES = ["lag1", "lag2", "past_mean", "lat", "lon", "month_sin", "month_cos"]


# ---------------------------------------------------------------- data
@st.cache_data
def make_demo_data(seed: int = 42) -> pd.DataFrame:
    """SYNTHETIC accident records around Ahilyanagar region (for demo only)."""
    rng = np.random.default_rng(seed)
    hotspots = [(19.77, 74.48, 5), (19.76, 74.00, 3), (19.55, 74.65, 3),
                (19.95, 74.20, 2), (19.30, 74.50, 2), (19.70, 74.30, 1.5)]
    months = pd.period_range("2023-01", "2025-12", freq="M")
    rows = []
    for m in months:
        n = rng.integers(70, 110)
        for _ in range(n):
            if rng.random() < 0.7:
                lat0, lon0, w = hotspots[rng.choice(len(hotspots), p=np.array([h[2] for h in hotspots]) / sum(h[2] for h in hotspots))]
                lat, lon = lat0 + rng.normal(0, 0.012), lon0 + rng.normal(0, 0.012)
            else:
                lat, lon = rng.uniform(19.2, 20.0), rng.uniform(73.9, 74.8)
            day = rng.integers(1, 28)
            hour = int(rng.choice(24, p=_hour_probs()))
            rows.append((pd.Timestamp(m.year, m.month, day, hour, rng.integers(0, 60)), lat, lon,
                         rng.choice(["Clear", "Rainy", "Foggy"], p=[.8, .13, .07]),
                         rng.choice(["Highway", "Urban", "Rural"], p=[.5, .3, .2]),
                         rng.choice(["Minor", "Serious", "Fatal"], p=[.6, .3, .1])))
    return pd.DataFrame(rows, columns=["datetime", "latitude", "longitude", "weather", "road_type", "severity"])


def _hour_probs():
    p = np.ones(24)
    p[[7, 8, 9, 17, 18, 19, 20]] = 2.5
    p[[0, 1, 2, 3, 4]] = 0.5
    return p / p.sum()


def guess(cols, keys):
    for k in keys:
        for c in cols:
            if k in c.lower():
                return c
    return None


@st.cache_data
def clean(df: pd.DataFrame, dt_col, lat_col, lon_col):
    log = [("Raw rows", len(df))]
    d = df.copy()
    d["datetime"] = pd.to_datetime(d[dt_col], errors="coerce")
    d["latitude"] = pd.to_numeric(d[lat_col], errors="coerce")
    d["longitude"] = pd.to_numeric(d[lon_col], errors="coerce")
    d = d.dropna(subset=["datetime", "latitude", "longitude"])
    log.append(("After removing missing date/coordinates", len(d)))
    d = d[d.latitude.between(-90, 90) & d.longitude.between(-180, 180)
          & ~((d.latitude == 0) & (d.longitude == 0))]
    log.append(("After removing invalid coordinates", len(d)))
    d = d.drop_duplicates(subset=["datetime", "latitude", "longitude"])
    log.append(("After removing duplicates", len(d)))
    d["hour"] = d.datetime.dt.hour
    d["weekday"] = d.datetime.dt.day_name()
    d["period"] = d.datetime.dt.to_period("M")
    return d.reset_index(drop=True), log


def build_panel(d: pd.DataFrame, cell_deg: float, min_total: int):
    d = d.copy()
    d["cell_lat"] = (d.latitude // cell_deg) * cell_deg + cell_deg / 2
    d["cell_lon"] = (d.longitude // cell_deg) * cell_deg + cell_deg / 2
    d["cell"] = d.cell_lat.round(4).astype(str) + "," + d.cell_lon.round(4).astype(str)
    keep = d.cell.value_counts()
    d = d[d.cell.isin(keep[keep >= min_total].index)]
    counts = d.groupby(["cell", "period"]).size().unstack(fill_value=0)
    periods = pd.period_range(counts.columns.min(), counts.columns.max(), freq="M")
    counts = counts.reindex(columns=periods, fill_value=0)
    centers = d.groupby("cell")[["cell_lat", "cell_lon"]].first()

    long = counts.stack().rename("count").reset_index()
    long.columns = ["cell", "period", "count"]
    long = long.sort_values(["cell", "period"])
    g = long.groupby("cell")["count"]
    long["lag1"] = g.shift(1)
    long["lag2"] = g.shift(2)
    long["past_mean"] = g.transform(lambda s: s.shift(1).expanding().mean())
    long = long.dropna().merge(centers, left_on="cell", right_index=True)
    long = long.rename(columns={"cell_lat": "lat", "cell_lon": "lon"})
    mnum = long.period.dt.month
    long["month_sin"], long["month_cos"] = np.sin(2 * np.pi * mnum / 12), np.cos(2 * np.pi * mnum / 12)
    return long.reset_index(drop=True), counts, centers


# ---------------------------------------------------------------- UI
st.title("🚧 Road Accident Risk Hotspot Prediction")
st.caption("PBL prototype · Linear Regression (accident counts) + Logistic Regression (high-risk label)")

with st.sidebar:
    st.header("1. Data")
    src = st.radio("Source", ["Demo data (synthetic)", "Upload CSV"])
    if src == "Upload CSV":
        up = st.file_uploader("Row-level accident CSV", type="csv")
        if up is None:
            st.info("Upload a CSV with a date/time column plus latitude and longitude.")
            st.stop()
        raw = pd.read_csv(up)
        cols = list(raw.columns)
        dt_col = st.selectbox("Date/time column", cols, index=cols.index(guess(cols, ["date", "time"])) if guess(cols, ["date", "time"]) else 0)
        lat_col = st.selectbox("Latitude column", cols, index=cols.index(guess(cols, ["lat"])) if guess(cols, ["lat"]) else 0)
        lon_col = st.selectbox("Longitude column", cols, index=cols.index(guess(cols, ["lon", "lng"])) if guess(cols, ["lon", "lng"]) else 0)
    else:
        raw, dt_col, lat_col, lon_col = make_demo_data(), "datetime", "latitude", "longitude"
        st.warning("Synthetic data: results are for demonstration only.")

    st.header("2. Grouping & labels")
    cell_km = st.slider("Grid cell size (km)", 1, 20, 5)
    min_total = st.slider("Min. accidents per cell (filter)", 1, 50, 10)
    quant = st.slider("High-risk = top quantile of TRAIN counts", 0.50, 0.95, 0.75, 0.05)
    test_periods = st.slider("Test period (last N months)", 2, 12, 6)

data, log = clean(raw, dt_col, lat_col, lon_col)
if len(data) < 50:
    st.error("Too few valid rows after cleaning.")
    st.stop()
cell_deg = cell_km / 111.0
panel, counts, centers = build_panel(data, cell_deg, min_total)
all_periods = sorted(panel.period.unique())

tab_data, tab_eda, tab_model, tab_map, tab_ref = st.tabs(
    ["📋 Data & cleaning", "📊 EDA", "🤖 Models", "🗺️ Hotspot map", "📚 Reference & limits"])

# ---- Data
with tab_data:
    c1, c2, c3 = st.columns(3)
    c1.metric("Clean records", f"{len(data):,}")
    c2.metric("Grid cells used", f"{counts.shape[0]:,}")
    c3.metric("Months covered", counts.shape[1])
    st.subheader("Cleaning log")
    st.dataframe(pd.DataFrame(log, columns=["Step", "Rows"]), hide_index=True, use_container_width=True)
    st.subheader("Preview")
    st.dataframe(data.head(200), use_container_width=True)

# ---- EDA
with tab_eda:
    a, b = st.columns(2)
    hr = data.hour.value_counts().sort_index().reset_index()
    hr.columns = ["hour", "accidents"]
    a.plotly_chart(px.bar(hr, x="hour", y="accidents", title="Accidents by hour of day"), use_container_width=True)
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    wd = data.weekday.value_counts().reindex(order).reset_index()
    wd.columns = ["weekday", "accidents"]
    b.plotly_chart(px.bar(wd, x="weekday", y="accidents", title="Accidents by day of week"), use_container_width=True)
    mo = data.groupby("period").size().reset_index(name="accidents")
    mo["period"] = mo.period.astype(str)
    st.plotly_chart(px.line(mo, x="period", y="accidents", markers=True, title="Monthly accident counts"), use_container_width=True)
    for col in [c for c in ["weather", "road_type", "severity"] if c in data.columns]:
        vc = data[col].value_counts().reset_index()
        vc.columns = [col, "accidents"]
        st.plotly_chart(px.bar(vc, x=col, y="accidents", title=f"Accidents by {col}"), use_container_width=True)
    st.subheader("Top grid cells by total accidents")
    top = counts.sum(axis=1).sort_values(ascending=False).head(15).reset_index()
    top.columns = ["cell (lat,lon)", "accidents"]
    st.dataframe(top, hide_index=True, use_container_width=True)

# ---- Models
with tab_model:
    if len(all_periods) <= test_periods + 3:
        st.error("Not enough months of data for this test-period size. Reduce the test period.")
        st.stop()
    cutoff = all_periods[-test_periods]
    train, test = panel[panel.period < cutoff], panel[panel.period >= cutoff]
    st.write(f"**Chronological split:** train = months before {cutoff} ({len(train):,} rows), "
             f"test = {cutoff} onward ({len(test):,} rows).")

    scaler = StandardScaler().fit(train[FEATURES])
    Xtr, Xte = scaler.transform(train[FEATURES]), scaler.transform(test[FEATURES])

    # Linear Regression
    lin = LinearRegression().fit(Xtr, train["count"])
    pred = np.clip(lin.predict(Xte), 0, None)
    st.subheader("Linear Regression - accidents per cell per month")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{mean_absolute_error(test['count'], pred):.3f}")
    m2.metric("RMSE", f"{np.sqrt(mean_squared_error(test['count'], pred)):.3f}")
    m3.metric("R²", f"{r2_score(test['count'], pred):.3f}")
    ap = pd.DataFrame({"Actual": test["count"].values, "Predicted": pred})
    fig = px.scatter(ap, x="Actual", y="Predicted", opacity=0.5, title="Actual vs predicted (test set)")
    mx = max(ap.Actual.max(), ap.Predicted.max())
    fig.add_shape(type="line", x0=0, y0=0, x1=mx, y1=mx, line=dict(dash="dash"))
    st.plotly_chart(fig, use_container_width=True)

    # Logistic Regression
    thr = max(1.0, float(train["count"].quantile(quant)))
    ytr, yte = (train["count"] >= thr).astype(int), (test["count"] >= thr).astype(int)
    st.subheader("Logistic Regression - high-risk vs lower-risk")
    st.write(f"High-risk rule (from training data only): **count ≥ {thr:.1f} accidents** in a cell-month. "
             f"Train high-risk share: {ytr.mean():.1%} · Test: {yte.mean():.1%}")
    if ytr.nunique() < 2:
        st.warning("Training data has only one class; lower the quantile or cell size.")
        log_model = None
    else:
        log_model = LogisticRegression(class_weight="balanced", max_iter=1000).fit(Xtr, ytr)
        proba = log_model.predict_proba(Xte)[:, 1]
        cut = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05)
        yp = (proba >= cut).astype(int)
        k = st.columns(4)
        k[0].metric("Accuracy", f"{accuracy_score(yte, yp):.3f}")
        k[1].metric("Precision", f"{precision_score(yte, yp, zero_division=0):.3f}")
        k[2].metric("Recall", f"{recall_score(yte, yp, zero_division=0):.3f}")
        k[3].metric("F1-score", f"{f1_score(yte, yp, zero_division=0):.3f}")
        cm = confusion_matrix(yte, yp, labels=[0, 1])
        st.plotly_chart(px.imshow(cm, text_auto=True, x=["Pred lower", "Pred high"], y=["Actual lower", "Actual high"],
                                  title="Confusion matrix (test set)", color_continuous_scale="Blues"),
                        use_container_width=True)
        if yte.nunique() < 2:
            st.caption("Test set contains only one class, so precision/recall/F1 may be 0 or undefined.")
    st.caption("Features use only past counts, location and month, so nothing from the predicted month leaks in.")

# ---- Map
with tab_map:
    st.subheader("Predicted hotspots for the next month")
    nxt = all_periods[-1] + 1
    full = panel
    Xall = StandardScaler().fit(full[FEATURES])
    lin_f = LinearRegression().fit(Xall.transform(full[FEATURES]), full["count"])
    thr_all = max(1.0, float(full["count"].quantile(quant)))
    yall = (full["count"] >= thr_all).astype(int)
    rows = []
    for cell, r in counts.iterrows():
        v = r.values
        rows.append(dict(cell=cell, lat=centers.loc[cell, "cell_lat"], lon=centers.loc[cell, "cell_lon"],
                         lag1=v[-1], lag2=v[-2], past_mean=v.mean(),
                         month_sin=np.sin(2 * np.pi * nxt.month / 12), month_cos=np.cos(2 * np.pi * nxt.month / 12)))
    nx = pd.DataFrame(rows)
    nx["predicted_count"] = np.clip(lin_f.predict(Xall.transform(nx[FEATURES])), 0, None).round(2)
    if yall.nunique() == 2:
        lg = LogisticRegression(class_weight="balanced", max_iter=1000).fit(Xall.transform(full[FEATURES]), yall)
        nx["high_risk_prob"] = lg.predict_proba(Xall.transform(nx[FEATURES]))[:, 1].round(3)
    else:
        nx["high_risk_prob"] = np.nan
    nx = nx.sort_values("predicted_count", ascending=False)
    topn = st.slider("Cells to show", 5, min(100, len(nx)), min(20, len(nx)))
    show = nx.head(topn)
    st.caption(f"Forecast month: {nxt} · models refit on all available months.")
    mapfn = getattr(px, "scatter_map", None) or px.scatter_mapbox
    kw = dict(lat="lat", lon="lon", size="predicted_count", color="high_risk_prob" if show.high_risk_prob.notna().all() else "predicted_count",
              hover_name="cell", zoom=8, height=520, color_continuous_scale="YlOrRd")
    fig = mapfn(show, **kw)
    fig.update_layout(map_style="open-street-map") if hasattr(px, "scatter_map") else fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(show[["cell", "predicted_count", "high_risk_prob", "lag1", "past_mean"]]
                 .rename(columns={"lag1": "last_month_count", "past_mean": "avg_monthly_count"}),
                 hide_index=True, use_container_width=True)
    st.download_button("Download hotspot table (CSV)", show.to_csv(index=False), "hotspots.csv", "text/csv")

# ---- Reference
with tab_ref:
    st.subheader("MoRTH national weather statistics (descriptive context)")
    ref = pd.DataFrame({
        "Weather": ["Sunny / Clear", "Rainy", "Foggy / Misty", "Hail / Sleet"],
        "Accidents 2023": [37457, 3771, 2212, 0], "Accidents 2024": [36506, 4228, 2184, 144],
        "Killed 2024": [10524, 1189, 649, 28], "Injured 2024": [44184, 5619, 2625, 163]})
    st.dataframe(ref, hide_index=True, use_container_width=True)
    st.plotly_chart(px.bar(ref, x="Weather", y=["Accidents 2023", "Accidents 2024"], barmode="group",
                           title="Reported accidents by weather condition (India)"), use_container_width=True)
    st.caption("Source: MoRTH, Road Accidents in India 2024, Table 3.8. National aggregates, not model outputs.")
    st.markdown("""
**Limitations**
- Weather totals ignore traffic exposure; clear-weather counts are high mainly because most travel is in clear weather.
- The high-risk label is a proxy based on historical counts, not a causal risk measure.
- Predictions support inspection and prioritisation; they should be validated by road-safety professionals.
- Demo data is synthetic. Report metrics only from your real, executed dataset.
""")
