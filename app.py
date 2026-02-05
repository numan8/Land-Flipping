import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Cash Sales Velocity Dashboard",
    layout="wide"
)

st.title("Cash Sales Velocity Dashboard")
st.caption("Goal: Find pricing/markup sweet spots that maximize probability of selling in ≤30 or ≤60 days, and increase compounding cycles.")

# ----------------------------
# Load data
# ----------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    needed = [
        "Acres",
        "County, State",
        "Property Location or City",
        "Total Purchase Price",
        "PURCHASE DATE",
        "SALE DATE - start",
        "Cash Sales Price - amount",
        "days_to_sale",
        "markup_multiple",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    # Type casting
    df["PURCHASE DATE"] = pd.to_datetime(df["PURCHASE DATE"], errors="coerce")
    df["SALE DATE - start"] = pd.to_datetime(df["SALE DATE - start"], errors="coerce")

    num_cols = ["Acres", "Total Purchase Price", "Cash Sales Price - amount", "days_to_sale", "markup_multiple"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Derived metrics
    df["profit_$"] = df["Cash Sales Price - amount"] - df["Total Purchase Price"]
    df["profit_pct_cost"] = np.where(df["Total Purchase Price"] > 0, df["profit_$"] / df["Total Purchase Price"], np.nan)

    # Commissions assumptions
    # - Acq Agent: 4% of PROFIT
    # - Sales Agent: 4% of PROFIT
    # - Affiliate/Listing Agent: 10% of SALE PRICE
    df["commission_profit_based_$"] = 0.08 * df["profit_$"].clip(lower=0)  # only if profit positive
    df["commission_sale_based_$"] = 0.10 * df["Cash Sales Price - amount"].clip(lower=0)

    df["net_profit_$"] = df["profit_$"] - df["commission_profit_based_$"] - df["commission_sale_based_$"]
    df["net_profit_pct_cost"] = np.where(df["Total Purchase Price"] > 0, df["net_profit_$"] / df["Total Purchase Price"], np.nan)

    # Buckets
    df["speed_bucket"] = pd.cut(
        df["days_to_sale"],
        bins=[-1, 30, 60, 90, 180, 3650],
        labels=["≤30", "31–60", "61–90", "91–180", "181+"]
    )

    df["acres_bucket"] = pd.cut(
        df["Acres"],
        bins=[-0.01, 0.25, 0.5, 1, 2, 5, 10, 1e9],
        labels=["≤0.25", "0.26–0.5", "0.51–1", "1.01–2", "2.01–5", "5.01–10", "10+"]
    )

    return df


DATA_PATH = "ai_stats_clean_for_velocity.csv"  # put this file next to app.py
df = load_data(DATA_PATH)

# ----------------------------
# Sidebar filters
# ----------------------------
st.sidebar.header("Filters")

min_date = df["PURCHASE DATE"].min()
max_date = df["PURCHASE DATE"].max()

date_range = st.sidebar.date_input(
    "Purchase date range",
    value=(min_date.date() if pd.notna(min_date) else None,
           max_date.date() if pd.notna(max_date) else None),
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
else:
    start_date, end_date = min_date, max_date

county_options = sorted([c for c in df["County, State"].dropna().unique()])
selected_counties = st.sidebar.multiselect("County, State", county_options, default=[])

city_options = sorted([c for c in df["Property Location or City"].dropna().unique()])
selected_cities = st.sidebar.multiselect("City", city_options, default=[])

dmin, dmax = int(df["days_to_sale"].min()), int(df["days_to_sale"].max())
days_range = st.sidebar.slider("Days to sale range", min_value=dmin, max_value=dmax, value=(dmin, dmax))

mm_min = float(np.nanmin(df["markup_multiple"]))
mm_max = float(np.nanmax(df["markup_multiple"]))
markup_range = st.sidebar.slider(
    "Markup multiple range (Sale / Total Cost)",
    min_value=float(mm_min),
    max_value=float(mm_max),
    value=(float(mm_min), float(mm_max))
)

# Apply filters
f = df.copy()
f = f[(f["PURCHASE DATE"] >= start_date) & (f["PURCHASE DATE"] <= end_date)]
if selected_counties:
    f = f[f["County, State"].isin(selected_counties)]
if selected_cities:
    f = f[f["Property Location or City"].isin(selected_cities)]
f = f[(f["days_to_sale"] >= days_range[0]) & (f["days_to_sale"] <= days_range[1])]
f = f[(f["markup_multiple"] >= markup_range[0]) & (f["markup_multiple"] <= markup_range[1])]

# ----------------------------
# Helper: sweet-spot table
# ----------------------------
def sweet_spot_table(data: pd.DataFrame, target_days: int = 30, n_bins: int = 8) -> pd.DataFrame:
    d = data.dropna(subset=["markup_multiple", "days_to_sale"]).copy()
    if len(d) < 10:
        return pd.DataFrame()

    try:
        d["mm_bin"] = pd.qcut(d["markup_multiple"], q=n_bins, duplicates="drop")
    except ValueError:
        d["mm_bin"] = pd.cut(d["markup_multiple"], bins=n_bins)

    g = d.groupby("mm_bin", observed=True).agg(
        n=("markup_multiple", "size"),
        median_markup=("markup_multiple", "median"),
        median_days=("days_to_sale", "median"),
        p_sell_within_target=("days_to_sale", lambda x: float(np.mean(x <= target_days))),
        median_net_profit_pct=("net_profit_pct_cost", "median"),
        median_net_profit_dollars=("net_profit_$", "median"),
    ).reset_index()

    g["mm_bin"] = g["mm_bin"].astype(str)
    g = g.sort_values("median_markup")

    # Score: velocity-first
    g["score"] = g["p_sell_within_target"] * 0.75 + g["median_net_profit_pct"].fillna(0) * 0.25

    # Display formatting
    g["p_sell_within_target"] = (g["p_sell_within_target"] * 100).round(1)
    g["median_net_profit_pct"] = (g["median_net_profit_pct"] * 100).round(1)
    g["median_net_profit_dollars"] = g["median_net_profit_dollars"].round(0)

    return g


# ----------------------------
# KPI row
# ----------------------------
col1, col2, col3, col4, col5, col6 = st.columns(6)

n = len(f)
pct_30 = (f["days_to_sale"] <= 30).mean() * 100 if n else 0
pct_60 = (f["days_to_sale"] <= 60).mean() * 100 if n else 0

median_days = float(np.nanmedian(f["days_to_sale"])) if n else np.nan
median_mm = float(np.nanmedian(f["markup_multiple"])) if n else np.nan
median_net = float(np.nanmedian(f["net_profit_pct_cost"])) * 100 if n else np.nan

cycles_est = (365 / median_days) if (n and pd.notna(median_days) and median_days > 0) else np.nan

col1.metric("Deals (filtered)", f"{n:,}")
col2.metric("Sell ≤30 days", f"{pct_30:.1f}%")
col3.metric("Sell ≤60 days", f"{pct_60:.1f}%")
col4.metric("Median days to sale", f"{median_days:.0f}" if pd.notna(median_days) else "—")
col5.metric("Median markup multiple", f"{median_mm:.2f}" if pd.notna(median_mm) else "—")
col6.metric("Est. cycles/year (median)", f"{cycles_est:.1f}" if pd.notna(cycles_est) else "—")

st.divider()

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Sweet Spot (≤30/≤60)", "Velocity & Pricing", "Geo / Property Segments", "Data Table"])

# ----------------------------
# Tab 1: Sweet Spot
# ----------------------------
with tab1:
    st.subheader("Markup sweet spot: probability of selling fast")

    left, right = st.columns(2)

    with left:
        st.markdown("### ≤30 days sweet spot")
        t30 = sweet_spot_table(f, target_days=30, n_bins=8)
        if t30.empty:
            st.info("Not enough rows in current filter to compute bins.")
        else:
            st.dataframe(
                t30[["mm_bin", "n", "median_markup", "median_days", "p_sell_within_target",
                     "median_net_profit_pct", "median_net_profit_dollars", "score"]],
                use_container_width=True
            )

            best30 = t30.sort_values("score", ascending=False).head(1)
            st.success(
                f"Best (≤30d) bin: **{best30['mm_bin'].iloc[0]}** | "
                f"Median markup **{best30['median_markup'].iloc[0]:.2f}x** | "
                f"Sell≤30 **{best30['p_sell_within_target'].iloc[0]:.1f}%** | "
                f"Median net margin **{best30['median_net_profit_pct'].iloc[0]:.1f}%**"
            )

    with right:
        st.markdown("### ≤60 days sweet spot")
        t60 = sweet_spot_table(f, target_days=60, n_bins=8)
        if t60.empty:
            st.info("Not enough rows in current filter to compute bins.")
        else:
            st.dataframe(
                t60[["mm_bin", "n", "median_markup", "median_days", "p_sell_within_target",
                     "median_net_profit_pct", "median_net_profit_dollars", "score"]],
                use_container_width=True
            )

            best60 = t60.sort_values("score", ascending=False).head(1)
            st.success(
                f"Best (≤60d) bin: **{best60['mm_bin'].iloc[0]}** | "
                f"Median markup **{best60['median_markup'].iloc[0]:.2f}x** | "
                f"Sell≤60 **{best60['p_sell_within_target'].iloc[0]:.1f}%** | "
                f"Median net margin **{best60['median_net_profit_pct'].iloc[0]:.1f}%**"
            )

    st.markdown("### Probability curve (Sell within X days vs markup multiple)")
    target_days = st.slider("Choose target days", 10, 120, 30, step=5)

    d = f.dropna(subset=["markup_multiple", "days_to_sale"]).copy()
    if len(d) >= 10:
        d = d.sort_values("markup_multiple")
        window = max(10, int(len(d) * 0.08))
        d["p_sell"] = (d["days_to_sale"] <= target_days).astype(int)
        d["p_sell_smooth"] = d["p_sell"].rolling(window=window, min_periods=max(5, window // 3)).mean()

        fig = px.line(
            d,
            x="markup_multiple",
            y="p_sell_smooth",
            labels={
                "markup_multiple": "Markup multiple (Sale / Total Cost)",
                "p_sell_smooth": f"Smoothed P(sell ≤ {target_days} days)"
            },
            title=f"Probability of Fast Sale vs Markup (window={window})"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough rows to plot probability curve.")

# ----------------------------
# Tab 2: Velocity & Pricing
# ----------------------------
with tab2:
    st.subheader("Velocity drivers and pricing behavior")

    c1, c2 = st.columns(2)

    with c1:
        fig = px.histogram(
            f.dropna(subset=["days_to_sale"]),
            x="days_to_sale",
            nbins=40,
            title="Distribution of Days to Sale"
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.histogram(
            f.dropna(subset=["markup_multiple"]),
            x="markup_multiple",
            nbins=40,
            title="Distribution of Markup Multiple (Sale / Total Cost)"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Markup vs Days to Sale (scatter)")
    d = f.dropna(subset=["markup_multiple", "days_to_sale"]).copy()
    if len(d) >= 3:
        fig = px.scatter(
            d,
            x="markup_multiple",
            y="days_to_sale",
            color="speed_bucket",
            hover_data=[
                "County, State",
                "Property Location or City",
                "Acres",
                "Total Purchase Price",
                "Cash Sales Price - amount",
                "net_profit_$",
                "net_profit_pct_cost",
            ],
            title="Markup Multiple vs Days to Sale (colored by speed bucket)"
        )
        st.plotly_chart(fig, use_container_width=True)

        corr = d[["markup_multiple", "days_to_sale"]].corr().iloc[0, 1]
        st.info(f"Correlation (markup_multiple vs days_to_sale): **{corr:.2f}** (directional only, not causation).")
    else:
        st.info("Not enough rows to build scatter plot.")

    st.markdown("### Profit vs Velocity tradeoff (net)")
    d2 = f.dropna(subset=["net_profit_pct_cost", "days_to_sale"]).copy()
    if len(d2) >= 3:
        fig = px.scatter(
            d2,
            x="net_profit_pct_cost",
            y="days_to_sale",
            color="speed_bucket",
            hover_data=["County, State", "Property Location or City", "Acres", "net_profit_$"],
            labels={"net_profit_pct_cost": "Net Profit % of Cost"},
            title="Net Profit % vs Days to Sale"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough rows to plot profit vs velocity.")

# ----------------------------
# Tab 3: Geo / Segments
# ----------------------------
with tab3:
    st.subheader("Which counties / cities / acreage ranges sell faster?")

    c1, c2 = st.columns(2)

    with c1:
        g = f.dropna(subset=["County, State", "days_to_sale"]).groupby("County, State", observed=True).agg(
            n=("days_to_sale", "size"),
            median_days=("days_to_sale", "median"),
            pct_30=("days_to_sale", lambda x: (x <= 30).mean() * 100),
            pct_60=("days_to_sale", lambda x: (x <= 60).mean() * 100),
            median_markup=("markup_multiple", "median"),
        ).reset_index()

        g = g[g["n"] >= 3].sort_values("median_days")

        fig = px.bar(
            g.head(20),
            x="median_days",
            y="County, State",
            orientation="h",
            hover_data=["n", "pct_30", "pct_60", "median_markup"],
            title="Top 20 Counties by Fastest Median Days (min 3 deals)"
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        a = f.dropna(subset=["acres_bucket", "days_to_sale"]).groupby("acres_bucket", observed=True).agg(
            n=("days_to_sale", "size"),
            median_days=("days_to_sale", "median"),
            pct_30=("days_to_sale", lambda x: (x <= 30).mean() * 100),
            pct_60=("days_to_sale", lambda x: (x <= 60).mean() * 100),
            median_markup=("markup_multiple", "median"),
        ).reset_index().sort_values("median_days")

        fig = px.bar(
            a,
            x="acres_bucket",
            y="median_days",
            hover_data=["n", "pct_30", "pct_60", "median_markup"],
            title="Median Days by Acres Bucket"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### City leaderboard (fastest median days)")
    city = f.dropna(subset=["Property Location or City", "days_to_sale"]).groupby("Property Location or City", observed=True).agg(
        n=("days_to_sale", "size"),
        median_days=("days_to_sale", "median"),
        pct_30=("days_to_sale", lambda x: (x <= 30).mean() * 100),
        pct_60=("days_to_sale", lambda x: (x <= 60).mean() * 100),
        median_markup=("markup_multiple", "median"),
    ).reset_index()

    city = city[city["n"] >= 3].sort_values("median_days")
    st.dataframe(city.head(30), use_container_width=True)

# ----------------------------
# Tab 4: Data table + export
# ----------------------------
with tab4:
    st.subheader("Filtered dataset")
    show_cols = [
        "County, State",
        "Property Location or City",
        "Acres",
        "Total Purchase Price",
        "Cash Sales Price - amount",
        "markup_multiple",
        "days_to_sale",
        "profit_$",
        "net_profit_$",
        "net_profit_pct_cost",
        "PURCHASE DATE",
        "SALE DATE - start",
    ]

    st.dataframe(f[show_cols].sort_values("SALE DATE - start", ascending=False), use_container_width=True)

    csv = f[show_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered CSV",
        data=csv,
        file_name="filtered_cash_sales_velocity.csv",
        mime="text/csv"
    )

st.caption("Note: This dashboard is descriptive. Prediction model can be added next (e.g., probability of selling ≤30/≤60 given markup, acres, county, etc.).")
