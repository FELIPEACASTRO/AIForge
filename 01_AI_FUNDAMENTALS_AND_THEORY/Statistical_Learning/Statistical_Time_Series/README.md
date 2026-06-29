# Statistical Time Series (ARIMA, GARCH, State Space)

> Classical, model-based methods for analyzing and forecasting data ordered in time, where observations are serially dependent and the goal is to model that dependence in the conditional mean (ARIMA/ETS), the conditional variance (ARCH/GARCH), or a latent dynamic state (state-space/Kalman).

## Why it matters

Time-ordered data — prices, demand, sensor streams, macroeconomic indicators — violate the i.i.d. assumption behind most ML, and ignoring autocorrelation produces overconfident, biased forecasts. Statistical time-series models remain the strong, interpretable baselines that deep forecasters must beat: they are sample-efficient, give calibrated prediction intervals, and expose structure (trend, seasonality, volatility clustering) directly. They are also the production default in finance (GARCH for risk), operations (ARIMA/ETS for demand), and engineering (Kalman filters for tracking and sensor fusion).

## Core concepts

- **Stochastic process & stationarity.** A series $\{y_t\}$ is (weakly) stationary if its mean, variance, and autocovariance $\gamma(k)=\mathrm{Cov}(y_t, y_{t-k})$ do not depend on $t$. Most theory assumes stationarity; non-stationary series are differenced or detrended first.
- **ACF/PACF.** The autocorrelation function $\rho(k)=\gamma(k)/\gamma(0)$ and partial autocorrelation guide model order: an AR($p$) cuts off in PACF at lag $p$; an MA($q$) cuts off in ACF at lag $q$.
- **AR, MA, ARMA.** AR($p$): $y_t = c + \sum_{i=1}^p \phi_i y_{t-i} + \varepsilon_t$. MA($q$): $y_t = \mu + \varepsilon_t + \sum_{j=1}^q \theta_j \varepsilon_{t-j}$. ARMA combines both. Using the lag operator $L$: $\phi(L)y_t = \theta(L)\varepsilon_t$.
- **Integration & ARIMA.** ARIMA($p,d,q$) applies ARMA to the $d$-times differenced series $(1-L)^d y_t$. SARIMA($p,d,q$)($P,D,Q$)$_s$ adds seasonal AR/MA/differencing at period $s$.
- **Unit roots & cointegration.** Tests (ADF, KPSS, Phillips–Perron) decide $d$; cointegration (Engle–Granger, Johansen) handles long-run equilibria among non-stationary series, enabling VECM.
- **Exponential smoothing / ETS.** Weighted averages with geometrically decaying weights; the ETS (Error, Trend, Seasonal) taxonomy (Hyndman et al.) gives a full state-space form, e.g. Holt–Winters for trend + seasonality.
- **Conditional heteroskedasticity (ARCH/GARCH).** Models time-varying variance / volatility clustering. ARCH($q$): $\sigma_t^2 = \omega + \sum \alpha_i \varepsilon_{t-i}^2$. GARCH($p,q$): $\sigma_t^2 = \omega + \sum_{i=1}^q \alpha_i \varepsilon_{t-i}^2 + \sum_{j=1}^p \beta_j \sigma_{t-j}^2$, fit by maximum likelihood.
- **State-space & Kalman filter.** A latent state evolves linearly with noise (transition: $x_t = F x_{t-1} + w_t$) and is observed noisily (measurement: $y_t = H x_t + v_t$). The Kalman filter gives the optimal recursive minimum-MSE estimate of $x_t$ under Gaussian noise; ARIMA and ETS can both be cast in this framework. Nonlinear/non-Gaussian extensions: EKF, UKF, particle filters.
- **Model selection & diagnostics.** AIC/BIC for order selection; Ljung–Box test on residuals for remaining autocorrelation; backtesting with rolling-origin (time-series) cross-validation rather than random splits.

## Algorithms / Methods

| Method | Models | Captures | Typical use |
|---|---|---|---|
| AR / MA / ARMA | Linear conditional mean of stationary series | Short-memory autocorrelation | Stationary signals |
| ARIMA | ARMA on differenced series ($d$) | Trend / non-stationarity | General univariate forecasting |
| SARIMA / SARIMAX | Seasonal ARIMA + exogenous regressors | Seasonality + covariates | Demand, traffic, retail |
| VAR / VECM | Multivariate AR / cointegration | Cross-series dynamics | Macro, multi-asset |
| Exponential smoothing (SES, Holt, Holt–Winters) | Weighted decay | Level, trend, seasonality | Fast, robust baselines |
| ETS (state-space) | Error–Trend–Seasonal taxonomy | Additive/multiplicative components | Automated forecasting |
| ARCH / GARCH | Conditional variance | Volatility clustering | Financial risk, VaR |
| EGARCH / GJR-GARCH / TGARCH | Asymmetric GARCH | Leverage effect | Equity/FX volatility |
| State-space + Kalman filter | Linear-Gaussian latent state | Smoothing, missing data, fusion | Tracking, nowcasting |
| EKF / UKF / particle filter | Nonlinear / non-Gaussian state | Nonlinear dynamics | Robotics, navigation |
| TBATS / Prophet | Trig seasonality / decomposable | Multiple/long seasonality | Business series with holidays |

## Tools & libraries

| Tool | Language | Focus | URL |
|---|---|---|---|
| statsmodels | Python | ARIMA, SARIMAX, VAR, ETS, state-space, unit-root tests | https://www.statsmodels.org/ |
| pmdarima | Python | `auto_arima` automatic order selection | https://alkaline-ml.com/pmdarima/ |
| arch | Python | ARCH/GARCH family, volatility, unit-root tests | https://arch.readthedocs.io/ |
| Prophet | Python/R | Decomposable additive forecasting | https://facebook.github.io/prophet/ |
| sktime | Python | Unified time-series ML/forecasting API | https://www.sktime.net/ |
| Darts | Python | Classical + deep forecasting, backtesting | https://unit8co.github.io/darts/ |
| StatsForecast (Nixtla) | Python | Fast ARIMA/ETS/Theta at scale | https://nixtlaverse.nixtla.io/statsforecast/ |
| pykalman | Python | Kalman filter & smoother, EM | https://pykalman.github.io/ |
| filterpy | Python | KF/EKF/UKF/particle filters | https://filterpy.readthedocs.io/ |
| forecast / fable (R) | R | ARIMA, ETS, TBATS (Hyndman) | https://pkg.robjhyndman.com/forecast/ |
| rugarch (R) | R | Univariate GARCH modeling | https://cran.r-project.org/package=rugarch |

## Learning resources

- **Forecasting: Principles and Practice (3rd ed.)** — Hyndman & Athanasopoulos. The definitive free online text on ARIMA/ETS forecasting. https://otexts.com/fpp3/ (Python version: https://otexts.com/fpppy/)
- **Time Series Analysis: Forecasting and Control** — Box, Jenkins, Reinsel & Ljung (5th ed., Wiley). The foundational ARIMA / Box–Jenkins reference. https://www.wiley.com/en-us/Time+Series+Analysis%3A+Forecasting+and+Control%2C+5th+Edition-p-9781118675021
- **Time Series Analysis and Its Applications: With R Examples** — Shumway & Stoffer. State-space and applied focus; companion site. https://www.stat.pitt.edu/stoffer/tsa4/
- **Penn State STAT 510: Applied Time Series Analysis** — free online course notes (ACF/PACF, ARIMA, GARCH). https://online.stat.psu.edu/stat510/
- **Box–Jenkins methodology overview** — Columbia Mailman public-health methods page. https://www.publichealth.columbia.edu/research/population-health-methods/box-jenkins-methodology
- **arch documentation** — practical GARCH modeling walkthroughs in Python. https://arch.readthedocs.io/en/latest/univariate/introduction.html
- **Kalman & Bayesian Filters in Python** — Roger Labbe. Free interactive Jupyter book on Kalman/EKF/UKF/particle filters. https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

## Key papers

- Box, G. E. P. & Jenkins, G. M. (1970). *Time Series Analysis: Forecasting and Control.* Holden-Day. (Established the ARIMA / Box–Jenkins methodology.) Current edition: https://www.wiley.com/en-us/Time+Series+Analysis%3A+Forecasting+and+Control%2C+5th+Edition-p-9781118675021
- Kalman, R. E. (1960). "A New Approach to Linear Filtering and Prediction Problems." *ASME J. Basic Engineering*, 82(1), 35–45. DOI: https://doi.org/10.1115/1.3662552
- Engle, R. F. (1982). "Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation." *Econometrica*, 50(4), 987–1007. DOI: https://doi.org/10.2307/1912773
- Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity." *Journal of Econometrics*, 31(3), 307–327. DOI: https://doi.org/10.1016/0304-4076(86)90063-1
- Dickey, D. A. & Fuller, W. A. (1979). "Distribution of the Estimators for Autoregressive Time Series with a Unit Root." *JASA*, 74(366a), 427–431. DOI: https://doi.org/10.1080/01621459.1979.10482531
- Engle, R. F. & Granger, C. W. J. (1987). "Co-integration and Error Correction: Representation, Estimation, and Testing." *Econometrica*, 55(2), 251–276. DOI: https://doi.org/10.2307/1913236
- Hyndman, R. J., Koehler, A. B., Snyder, R. D. & Grose, S. (2002). "A State Space Framework for Automatic Forecasting Using Exponential Smoothing Methods." *Int. J. Forecasting*, 18(3), 439–454. DOI: https://doi.org/10.1016/S0169-2070(01)00110-8

## Cross-references in AIForge

- [Bayesian and Probabilistic ML](../../Bayesian_and_Probabilistic_ML/) — state-space models, Kalman filtering, and probabilistic forecasting.
- [State Space Models](../../State_Space_Models/) — modern deep sequence models (S4, Mamba) that descend from classical state-space theory.
- [Model Evaluation](../../Model_Evaluation/) — rolling-origin / time-series cross-validation and forecast accuracy metrics.
- [Optimization Algorithms](../../Optimization_Algorithms/) — maximum-likelihood and EM estimation underlying GARCH and Kalman fitting.

## Sources

- Forecasting: Principles and Practice — https://otexts.com/fpp3/
- Box–Jenkins methodology (Columbia) — https://www.publichealth.columbia.edu/research/population-health-methods/box-jenkins-methodology
- Box, Jenkins, Reinsel & Ljung, *Time Series Analysis* (Wiley) — https://www.wiley.com/en-us/Time+Series+Analysis%3A+Forecasting+and+Control%2C+5th+Edition-p-9781118675021
- Kalman (1960), DOI 10.1115/1.3662552 — https://asmedigitalcollection.asme.org/fluidsengineering/article/82/1/35/397706
- Engle (1982), Econometrica — https://www.econometricsociety.org/publications/econometrica/1982/07/01/autoregressive-conditional-heteroscedasticity-estimates
- Bollerslev (1986), Journal of Econometrics — https://www.sciencedirect.com/science/article/abs/pii/0304407686900631
- statsmodels documentation — https://www.statsmodels.org/
- arch (Kevin Sheppard) — https://github.com/bashtage/arch and https://arch.readthedocs.io/
- StatsForecast (Nixtla) — https://nixtlaverse.nixtla.io/statsforecast/
- Penn State STAT 510 — https://online.stat.psu.edu/stat510/
- Kalman and Bayesian Filters in Python (R. Labbe) — https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
