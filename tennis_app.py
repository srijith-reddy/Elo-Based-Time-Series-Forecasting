# ===========================================================
# 🎾 ATP ELO FORECAST DASHBOARD — Full Version
# ===========================================================
import os, glob, warnings, itertools
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy import signal
from scipy import stats
from joblib import Parallel, delayed
import itertools
warnings.filterwarnings("ignore")
st.set_page_config(page_title="🎾 ATP Elo Forecast Dashboard", layout="wide")


if st.button("♻️ Clear all cached data"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("✅ Cache cleared. Rerun the app to rebuild everything.")

# ===========================================================
# LOAD ATP DATA
# ===========================================================
@st.cache_data
def load_data():
    files = sorted(glob.glob(os.path.join("tennis_atp", "atp_matches_20*.csv")))
    if not files:
        st.error("❌ No match CSVs found in 'tennis_atp' folder.")
        st.stop()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["winner_name", "loser_name", "tourney_date"])
    df["surface"] = df["surface"].fillna("Unknown").str.title()
    surface_map = {"Hard": "H", "Clay": "C", "Grass": "G", "Carpet": "O", "Unknown": "UNK"}
    df["surface_code"] = df["surface"].map(surface_map).fillna("UNK")
    return df[(df["tourney_date"].dt.year >= 2000) & (df["tourney_date"].dt.year <= 2025)]

df = load_data()
st.success(f"✅ Loaded {len(df):,} matches ({df['tourney_date'].min().date()} → {df['tourney_date'].max().date()})")

# ===========================================================
# COMPUTE ADVANCED ELO
# ===========================================================
@st.cache_data
def compute_elo(df, base_elo=1500, K_base=32):
    level_weights = {"G": 1.25, "M": 1.1, "A": 1.0}
    elo = defaultdict(lambda: {"H": base_elo, "C": base_elo, "G": base_elo, "O": base_elo, "UNK": base_elo})
    winners, losers = [], []
    df = df.sort_values("tourney_date").reset_index(drop=True)
    for _, row in df.iterrows():
        w, l = row["winner_name"], row["loser_name"]
        surf = row.get("surface_code", "UNK")
        w_elo, l_elo = elo[w][surf], elo[l][surf]
        expected_w = 1 / (1 + 10 ** ((l_elo - w_elo) / 400))
        K = K_base * np.exp(-(df["tourney_date"].max() - row["tourney_date"]).days / 3650)
        if row.get("best_of", 3) == 5:
            K *= 1.2
        elo[w][surf] = w_elo + K * (1 - expected_w)
        elo[l][surf] = l_elo + K * (0 - (1 - expected_w))
        winners.append(w_elo)
        losers.append(l_elo)
    df["winner_elo"], df["loser_elo"] = winners, losers
    return df

df = compute_elo(df)
st.success("✅ Advanced Elo ratings computed successfully.")

# ===========================================================
# BUILD MONTHLY ELO SERIES
# ===========================================================
@st.cache_data
def build_monthly_elo(df):
    players = pd.unique(df[["winner_name", "loser_name"]].values.ravel("K"))
    series = {}
    for p in players:
        p_df = df[(df["winner_name"] == p) | (df["loser_name"] == p)]
        if p_df.empty:
            continue
        p_df["player_elo"] = np.where(p_df["winner_name"] == p, p_df["winner_elo"], p_df["loser_elo"])
        s = p_df.set_index("tourney_date").resample("M")["player_elo"].mean().interpolate()
        if len(s) >= 12:
            series[p] = s

    return series

player_monthly_elo = build_monthly_elo(df)
players = sorted(player_monthly_elo.keys())
st.success(f"✅ Built monthly Elo time series for {len(players)} players.")

# ===========================================================
# AUTO DIFFERENCING LOGIC (ADF-based — Smarter + Trend-aware)
# ===========================================================
def auto_diff_level(y, max_diff=2, sig=0.05):
    """
    Automatically determine differencing order (d) and whether a linear trend should be added.

    Logic:
      • Uses sequential ADF tests (up to max_diff) for stationarity
      • Detects deterministic trend via Pearson correlation with time index
      • Prevents over-differencing for short or trending series

    Returns:
        d (int): differencing order (0–2)
        trend_flag (bool): True if deterministic trend detected
    """
    y = pd.Series(y).dropna()

    # --- Step 1: Sequential ADF test ---
    for d in range(max_diff + 1):
        test_series = y.diff(d).dropna() if d > 0 else y
        try:
            pval = adfuller(test_series, autolag="AIC")[1]
            if pval < sig:
                break  # Stationarity achieved
        except Exception:
            continue
    else:
        d = max_diff

    # --- Step 2: Deterministic trend check ---
    try:
        t = np.arange(len(y))
        corr = abs(stats.pearsonr(t[-len(test_series):], test_series)[0]) if len(test_series) > 10 else 0
    except Exception:
        corr = 0
    trend_flag = corr > 0.25  # slightly more sensitive than 0.3

    # --- Step 3: Prevent over-differencing ---
    if len(y) < 80 and d > 1:
        d = 1
    if d == 2 and trend_flag:
        d = 1

    return d, trend_flag


# ===========================================================
# SMART MODEL SELECTION (Streamlit-safe)
# ===========================================================
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import ConvergenceWarning


def smart_evaluate_models(y, p_values, q_values):
    """
    Streamlit-safe ARIMA model evaluator.
    Performs:
      - Auto differencing and trend detection
      - Full grid search for (p, q)
      - Stability + Ljung–Box filtering
      - Smart composite scoring (parsimonious + interpretable)
    """

    # 🧩 Consistency & numerical stability
    np.random.seed(42)
    y = pd.Series(y).dropna().round(3)

    results = []

    # Auto differencing and trend detection
    d, trend_flag = auto_diff_level(y)
    trend = 't' if trend_flag else 'n'

    for p in p_values:
        for q in q_values:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                try:
                    fit = ARIMA(y, order=(p, d, q), trend=trend).fit()

                    # Residuals
                    resid = fit.resid.dropna()

                    # Residual diagnostics
                    lb_p = acorr_ljungbox(resid, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
                    in_rmse = round(np.sqrt(np.mean(resid ** 2)), 3)

                    # Stability checks
                    converged = fit.mle_retvals.get("converged", True)
                    warned = any(isinstance(x.message, ConvergenceWarning) for x in w)
                    if not converged or warned:
                        continue

                    # Round model metrics to reduce floating jitter
                    results.append({
                        "p": p,
                        "d": d,
                        "q": q,
                        "trend": trend,
                        "params": len(fit.params),
                        "AIC": round(fit.aic, 3),
                        "BIC": round(fit.bic, 3),
                        "RMSE_in": in_rmse,
                        "LB_p(10)": round(lb_p, 3),
                    })

                except Exception:
                    continue

    # Convert to DataFrame
    df = pd.DataFrame(results)
    if df.empty:
        print(f"⚠️ No models fit successfully for d={d}, trend={trend}")
        return df

    # Ljung–Box sanity filter
    df = df[df["LB_p(10)"] > 0.05].copy()
    if df.empty:
        print(f"⚠️ All models filtered out by Ljung–Box (p ≤ 0.05)")
        return df

    # Ranking system
    df["AIC_rank"] = df["AIC"].rank(ascending=True)
    df["BIC_rank"] = df["BIC"].rank(ascending=True)
    df["params_rank"] = df["params"].rank(ascending=True)
    df["RMSE_rank"] = df["RMSE_in"].rank(ascending=True)
    df["LB_rank"] = df["LB_p(10)"].rank(ascending=False)

    # Smart composite score
    df["SmartScore"] = (
        0.40 * df["AIC_rank"]
        + 0.20 * df["BIC_rank"]
        + 0.10 * df["params_rank"]
        + 0.25 * df["RMSE_rank"]
        + 0.05 * df["LB_rank"]
    )

    # Deterministic sorting (prevents random tie-flips)
    df = df.sort_values(["SmartScore", "AIC", "BIC", "params"]).reset_index(drop=True)

    return df.round(3)


# ===========================================================
# FIT BEST MODEL (Streamlit-safe, clean, and stable)
# ===========================================================
def fit_best_model(y):
    """
    Fits the best ARIMA model based on SmartScore ranking.
    Uses fallback ARIMA(1,1,0) if no stable models are found.
    """

    # 🔧 Define search grid here
    p_values = [0, 1, 2, 3, 6, 12]
    q_values = [0, 1, 2, 3, 6, 12]

    # Evaluate grid
    grid = smart_evaluate_models(y, p_values=p_values, q_values=q_values)

    if grid.empty:
        print("⚠️ No stable models found — fallback to ARIMA(1,1,0).")
        fallback = ARIMA(y, order=(1, 1, 0)).fit()
        meta = pd.Series({
            "p": 1,
            "d": 1,
            "q": 0,
            "trend": "n",
            "AIC": fallback.aic,
            "RMSE_in": np.nan
        })
        return fallback, meta

    # Select best model (deterministic)
    best = grid.iloc[0]
    model = ARIMA(
        y,
        order=(int(best.p), int(best.d), int(best.q)),
        trend=None if best.trend == "n" else "t"
    ).fit()

    return model, best


# ===========================================================
# STEP 7: 12-Month Forecasting (Per Player – Streamlit Ready)
# ===========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA

def forecast_single_player(player, player_monthly_elo, steps=12):
    """
    Forecast 12-month Elo ratings for ONE player using Smart ARIMA models.
    Includes out-of-sample RMSE via train/test split.
    Streamlit-friendly: interactive plots + results table.
    """
    y = player_monthly_elo[player].dropna()

    # --- Skip if series too short ---
    if len(y) < steps * 2:
        st.warning(f"⚠️ Not enough data to forecast {player}.")
        return None, None

    # --- 1️⃣ Train/Test Split ---
    split_idx = len(y) - steps
    train, test = y.iloc[:split_idx], y.iloc[split_idx:]

    # --- 2️⃣ Fit best model on training data ---
    model_train, meta_train = fit_best_model(train)

    # --- 3️⃣ Forecast on test window (pseudo-future) ---
    fc_test = model_train.get_forecast(steps=len(test))
    pred_test = fc_test.predicted_mean
    rmse_oos = np.sqrt(np.mean((test.values - pred_test.values) ** 2))

    # --- 4️⃣ Fit final model on full data ---
    final_model, meta_final = fit_best_model(y)
    fc_future = final_model.get_forecast(steps=steps)
    mean_forecast = fc_future.predicted_mean
    conf_int = fc_future.conf_int(alpha=0.05)

    # --- 5️⃣ Combine forecast values ---
    forecast_df = pd.DataFrame({
        "Month_Ahead": np.arange(1, steps + 1),
        "Forecast_Elo": mean_forecast.values,
        "Lower_CI": conf_int.iloc[:, 0].values,
        "Upper_CI": conf_int.iloc[:, 1].values,
        "RMSE_out": rmse_oos
    })

    # --- 6️⃣ Meta summary ---
    meta_summary_df = pd.DataFrame([{
        "Player": player,
        "Order": (int(meta_final['p']), int(meta_final['d']), int(meta_final['q'])),
        "Trend": meta_final.get("trend", "n"),
        "AIC": round(meta_final["AIC"], 2),
        "BIC": round(meta_final["BIC"], 2),
        "RMSE_out": round(rmse_oos, 2)
    }])

    # --- 7️⃣ Plot Forecast (Streamlit-ready) ---
    hist_idx = np.arange(len(y))
    fut_idx = np.arange(len(y), len(y) + steps)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(hist_idx, y.values, color="steelblue", lw=1.4, label="Historical Elo")
    ax.plot(fut_idx, mean_forecast, color="darkorange", lw=1.8, label="Forecast")
    ax.fill_between(fut_idx, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                    color="orange", alpha=0.25, label="95% CI")
    ax.set_title(f"{player} — 12-Month Elo Forecast", fontsize=11)
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Elo Rating")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    # --- 8️⃣ Show Results ---
    st.subheader(f"📊 {player} Forecast Summary")
    st.dataframe(meta_summary_df)

    st.subheader(f"📈 {player} 12-Month Forecast Data")
    st.dataframe(forecast_df)

    return forecast_df, meta_summary_df, final_model


def elo_win_prob(ea, eb):
    """
    Compute probability that Player A (rating ea) beats Player B (rating eb)
    using the standard Elo logistic formula.
    """
    return 1 / (1 + 10 ** (-(ea - eb) / 400))

# ===========================================================
# RESIDUAL DIAGNOSTICS
# ===========================================================
def residual_diagnostics(result, player):
    resid = result.resid.dropna()
    lb = acorr_ljungbox(resid, lags=[10, 20], return_df=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(resid, color="steelblue")
    axes[0].axhline(0, ls="--", c="gray")
    axes[0].set_title(f"{player} — Residuals Over Time")
    plot_acf(resid, lags=20, ax=axes[1], color="coral")
    axes[1].set_title("Residual ACF")
    plt.tight_layout()
    st.pyplot(fig)
    st.write("**Ljung–Box Test (H₀: No autocorrelation)**")
    st.dataframe(lb)

# ===========================================================
# STREAMLIT UI — FORECAST DASHBOARD (FIXED)
# ===========================================================
st.header("🎾 ATP Elo Forecast Dashboard")

# --- Player selection ---
col1, col2 = st.columns(2)
player_a = col1.selectbox(
    "Player A", players,
    index=players.index("Novak Djokovic") if "Novak Djokovic" in players else 0
)
player_b = col2.selectbox(
    "Player B", players,
    index=players.index("Carlos Alcaraz") if "Carlos Alcaraz" in players else 1
)
horizon = st.slider("Forecast Horizon (months)", 6, 24, 12)

# --- Forecast both players ---
with st.spinner(f"Running ARIMA model selection for {player_a} and {player_b}..."):
    fc_a, meta_a, model_a = forecast_single_player(player_a, player_monthly_elo, horizon)
    fc_b, meta_b, model_b = forecast_single_player(player_b, player_monthly_elo, horizon)

# ===========================================================
# MODEL SUMMARY (Fixed for DataFrame meta objects)
# ===========================================================
st.subheader("📊 Model Selection Summary")

if meta_a is not None and meta_b is not None:
    a_order = meta_a.iloc[0]["Order"]
    b_order = meta_b.iloc[0]["Order"]

    a_aic = meta_a.iloc[0]["AIC"]
    b_aic = meta_b.iloc[0]["AIC"]

    a_rmse = meta_a.iloc[0]["RMSE_out"]
    b_rmse = meta_b.iloc[0]["RMSE_out"]

    col1, col2 = st.columns(2)
    col1.markdown(f"**{player_a}** → ARIMA{a_order} | AIC={a_aic:.1f} | RMSE={a_rmse:.2f}")
    col2.markdown(f"**{player_b}** → ARIMA{b_order} | AIC={b_aic:.1f} | RMSE={b_rmse:.2f}")
else:
    st.warning("⚠️ Could not compute model summaries for one or both players.")

# ===========================================================
# MONTHLY ELO TREND
# ===========================================================
st.subheader("📈 Monthly Elo Trend (Selected Players)")
fig, ax = plt.subplots(figsize=(9, 4))
for player in [player_a, player_b]:
    series = player_monthly_elo[player].dropna()
    ax.plot(series.index, series.values, linewidth=2, label=player)
ax.set_title("Monthly Elo Ratings (2000–2025)")
ax.set_xlabel("Date")
ax.set_ylabel("Elo Rating")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)

# ===========================================================
# ACF / PACF OF AUTO-DIFFERENCED SERIES
# ===========================================================
st.subheader("🔍 ACF & PACF — Auto-Differenced Series (All Players)")

for player in [player_a, player_b]:
    y = player_monthly_elo[player].dropna()

    # --- Auto differencing + detrend detection ---
    d, trend_flag = auto_diff_level(y)

    # --- Apply transformation ---
    if d == 0:
        y_trans = y
        diff_label = "Original Series (d=0)"
    elif d == 1 and not trend_flag:
        y_trans = y.diff().dropna()
        diff_label = "First Difference (d=1)"
    elif d == 1 and trend_flag:
        y_d1 = y.diff().dropna()
        y_trans = pd.Series(signal.detrend(y_d1.values), index=y_d1.index)
        diff_label = "First Diff + Detrend (d=1 + trend)"
    else:
        y_trans = y.diff().diff().dropna()
        diff_label = "Second Difference (d=2)"

    # --- Plot ACF/PACF ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{player} — ACF & PACF ({diff_label})", fontsize=13, y=1.03)

    plot_acf(y_trans, lags=20, ax=axes[0], color="royalblue")
    axes[0].set_title("Autocorrelation (ACF)")
    axes[0].grid(alpha=0.3)

    plot_pacf(y_trans, lags=20, ax=axes[1], color="darkorange", method="ywm")
    axes[1].set_title("Partial Autocorrelation (PACF)")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    st.caption(f"🧩 {player}: Selected {diff_label}")

# ===========================================================
# RESIDUAL DIAGNOSTICS (Now uses actual model objects)
# ===========================================================
st.subheader("🧩 Residual Diagnostics (Selected Players)")
for player, model in zip([player_a, player_b], [model_a, model_b]):
    st.markdown(f"**{player} — Residual Diagnostics**")
    residual_diagnostics(model, player)

# ===========================================================
# FORECASTED WIN PROBABILITIES
# ===========================================================
st.subheader(f"🏆 {player_a} vs {player_b} — Forecasted Win Probabilities")

if fc_a is not None and fc_b is not None:
    fc = fc_a.copy()
    fc["Elo_A"], fc["Elo_B"] = fc_a["Forecast_Elo"], fc_b["Forecast_Elo"]
    fc["P(A_wins)"] = elo_win_prob(fc["Elo_A"], fc["Elo_B"])
    fc["P(B_wins)"] = 1 - fc["P(A_wins)"]

    st.dataframe(fc[["Month_Ahead", "Elo_A", "Elo_B", "P(A_wins)", "P(B_wins)"]].round(3))

    # --- Plot probabilities ---
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(fc["Month_Ahead"], fc["P(A_wins)"], marker="o", color="orange", label=f"P({player_a} wins)")
    ax1.plot(fc["Month_Ahead"], fc["P(B_wins)"], marker="o", color="steelblue", linestyle="--", label=f"P({player_b} wins)")
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Months Ahead")
    ax1.set_ylabel("Win Probability")
    ax1.legend()
    ax1.grid(alpha=0.3)
    st.pyplot(fig1)

    # --- Plot Elo forecasts side-by-side ---
    st.subheader("📈 Forecasted Elo Ratings")
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(fc_a["Month_Ahead"], fc_a["Forecast_Elo"], label=player_a, color="orange")
    ax2.fill_between(fc_a["Month_Ahead"], fc_a["Lower_CI"], fc_a["Upper_CI"], color="orange", alpha=0.2)
    ax2.plot(fc_b["Month_Ahead"], fc_b["Forecast_Elo"], label=player_b, color="steelblue")
    ax2.fill_between(fc_b["Month_Ahead"], fc_b["Lower_CI"], fc_b["Upper_CI"], color="steelblue", alpha=0.2)
    ax2.set_xlabel("Months Ahead")
    ax2.set_ylabel("Elo Rating")
    ax2.legend()
    st.pyplot(fig2)
else:
    st.warning("⚠️ Forecast data missing — check previous steps.")
