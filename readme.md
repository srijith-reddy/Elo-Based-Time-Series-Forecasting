# 🎾 Elo-Based Time Series Forecasting


An interactive ATP Elo Forecast Dashboard built with Python and Streamlit.  
This app automatically computes advanced surface-aware Elo ratings, constructs monthly time series for every ATP player, and applies a Smart ARIMA Model Selection system to forecast each player’s Elo trajectory up to 24 months ahead.  
It also visualizes win probabilities, residual diagnostics, and model interpretability plots (ACF/PACF) — giving fans, analysts, and researchers a statistical edge in understanding player form trends.

---

## 📊 Data Source

**Dataset:** [tennis_atp by Jeff Sackmann](https://github.com/JeffSackmann/tennis_atp)

This open-source database provides:
1. **ATP player, ranking, and match data (1968–present)** — including biographical info, rankings, and ranking points.  
2. **Tour-level results and match stats** — from 1991 for main draws, 2008 for challengers, and 2011 for qualifying.  
3. **Doubles and Davis Cup data** — tour-level doubles available since 2000.  

> Please review and comply with Jeff Sackmann’s [license terms](https://github.com/JeffSackmann/tennis_atp#attention).  
> If used for academic or research purposes, **cite the dataset** appropriately.

---

## 🧠 Model & Methodology

- **ARIMA Auto-Selection** — automated differencing and trend detection with ADF-based stationarity tests.  
- **SmartScore Ranking** — custom composite metric combining AIC, BIC, RMSE, and Ljung–Box stability for robust model selection.  
- **Dynamic Elo System** — continuously updated surface-adjusted ratings per player (Hard, Clay, Grass, Carpet).  
- **Forecast Horizon:** Configurable from 6 to 24 months.  
- **Diagnostics:** Residual ACF/PACF, Ljung–Box tests, and forecast uncertainty intervals.

### 🎯 Win Probability Formula

$$
P(A \text{ beats } B) = \frac{1}{1 + 10^{-\frac{(E_A - E_B)}{400}}}
$$

---

## 🚀 Run Locally

Clone this repository and launch the Streamlit dashboard on your system:

```bash
git clone https://github.com/srijith-reddy/Elo-Based-Time-Series-Forecasting.git
cd Elo-Based-Time-Series-Forecasting
pip install -r requirements.txt
streamlit run tennis_app.py
