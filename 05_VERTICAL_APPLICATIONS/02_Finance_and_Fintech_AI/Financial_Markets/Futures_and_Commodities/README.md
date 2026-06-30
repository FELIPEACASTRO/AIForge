# Futures & Commodities

> Futures are standardized exchange-traded contracts to buy/sell an asset at a set price on a future date. Commodities (energy, metals, agriculture) trade largely via futures. This page covers the term structure, roll mechanics, and ML for trend/commodity strategies.

## Core concepts

- **Futures mechanics**: standardized contracts, margin (initial/maintenance), mark-to-market, daily settlement, physical vs. cash settlement, expiry/rollover.
- **Term structure**: **contango** (forward > spot) vs. **backwardation** (forward < spot); the **roll yield** that dominates long-horizon returns.
- **Commodity sectors**: energy (WTI/Brent crude, natural gas), metals (gold, silver, copper), agriculture (corn, wheat, soybeans), softs (coffee, sugar).
- **Venues**: CME Group (CME/CBOT/NYMEX/COMEX), ICE, Eurex, LME, B3.
- **Players**: hedgers (producers/consumers), speculators, and **CTAs / managed futures** (systematic trend followers).

## Strategies & ML

| Strategy | Idea | ML angle |
|---|---|---|
| Trend following / time-series momentum | Ride persistent trends across many markets | ML trend filters; Moskowitz-Ooi-Pedersen TSMOM |
| Carry | Earn roll yield from term structure | ML on curve shape / inventory |
| Mean reversion / spread (calendar, crack, crush) | Relative value between contracts | Cointegration + ML |
| Macro / fundamental nowcasting | Supply-demand, inventories, weather | Satellite imagery, weather models, alt-data |
| Seasonality | Calendar effects in ags/energy | Feature engineering + GBDT |

ML uses: **satellite/geospatial data** (crop yields, oil-storage tank levels, shipping), **weather models** for ag/energy, demand nowcasting, and sequence models for volatility. Watch for **roll/continuation-contract construction** bias in backtests and thin-liquidity contracts.

## Data & tools

- Continuous-contract data (back-adjusted) from Nasdaq Data Link (Quandl), CME DataMine, Norgate, `yfinance` (limited). Satellite: Orbital Insight, Planet. Weather: NOAA.

## Key references

- Moskowitz, Ooi, Pedersen, *Time Series Momentum* (JFE 2012). https://www.sciencedirect.com/science/article/abs/pii/S0304405X11002613
- Lim, Zohren, Roberts, *Enhancing Time Series Momentum Strategies Using Deep Neural Networks* (2019). https://arxiv.org/abs/1904.04912
- Gorton & Rouwenhorst, *Facts and Fantasies about Commodity Futures* (2006).

## 🛢️ Commodity & energy data
- [Commodities & Energy Data Sources](./Commodities_and_Energy_Data_Sources.md) — EIA, USDA WASDE, CFTC COT, Baker Hughes, LME, Kpler/Vortexa (tankers), NOAA weather, satellite; 🇧🇷 CEPEA/ESALQ, ANP, ONS/CCEE.

## Related in AIForge
- [Algorithmic & Quant Trading](../Algorithmic_and_Quant_Trading/) · [Options & Derivatives](../Options_and_Derivatives/) · [Alternative Data & Sentiment](../Alternative_Data_and_Sentiment_Analysis/) · [Risk Management](../Risk_Management_and_Derivatives_Pricing/)
- Fundamentals: [`../../../../01_AI_FUNDAMENTALS_AND_THEORY/Statistical_Learning/`](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Statistical_Learning/)

**Keywords:** futures trading, commodities, term structure, contango backwardation, roll yield, CTA managed futures, time series momentum, deep learning trend following, satellite alternative data.
