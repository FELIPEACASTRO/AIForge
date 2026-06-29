# Fixed Income & Bonds

> Bonds are debt instruments paying coupons and principal. Fixed income is the largest asset class by value. This page covers yield curves, rates risk, credit, and ML for rates/credit modeling.

## Core concepts

- **Instruments**: government bonds (US Treasuries, Bunds, JGBs, Brazilian NTN/LTN), corporates, munis, MBS/ABS, inflation-linked (TIPS), money-market instruments.
- **Pricing & risk**: price-yield relationship, **duration** (rate sensitivity), **convexity**, DV01/PV01, accrued interest.
- **The yield curve**: term structure of interest rates; shapes (normal, flat, inverted); spot vs. forward rates; bootstrapping; nominal vs. real.
- **Credit**: credit spreads, ratings, default/recovery, CDS, credit migration.
- **Rates models**: short-rate (Vasicek, CIR, Hull-White), HJM, LIBOR/SOFR market models; curve construction.

## ML applications

| Task | Approach | Notes |
|---|---|---|
| Yield-curve modeling/forecasting | Nelson-Siegel(-Svensson), dynamic NS, **deep learning** | NN/RNN forecast curve factors (level/slope/curvature) |
| Credit-spread & default prediction | GBDT, survival models, GNN | Alternative-data credit; corporate distress |
| Credit ratings replication | Classification/ordinal regression | ML mimics/anticipates agency ratings |
| Liquidity & TRACE analytics | ML on transaction prints | Sparse, OTC-traded bonds; pricing illiquid names |
| Macro nowcasting (rates) | ML on macro releases | Central-bank policy, inflation, employment |
| MBS prepayment modeling | ML on borrower/loan data | Classic high-value application |

Challenges: bonds trade **OTC and infrequently** (sparse, stale prices), huge heterogeneous universes (many CUSIPs per issuer), and regime sensitivity to central-bank policy. **SOFR transition** (post-LIBOR) reshaped curve construction.

## Data & tools

- FRED (free macro/rates), US Treasury, ECB, TRACE (corporate bond trades), Bloomberg/Refinitiv (institutional). QuantLib for curve building & bond pricing.

## Key references

- Diebold & Li, *Forecasting the Term Structure of Government Bond Yields* (2006). https://www.sciencedirect.com/science/article/abs/pii/S0304407605000795
- Nelson & Siegel (1987); Svensson (1994) — curve parameterization.
- Bianchi, Büchner, Tamoni, *Bond Risk Premia with Machine Learning* (RFS 2021). https://academic.oup.com/rfs/article/34/2/1046/5821387

## Related in AIForge
- [Risk Management & Derivatives Pricing](../Risk_Management_and_Derivatives_Pricing/) · [Foreign Exchange (FX)](../Foreign_Exchange_FX/) · [Portfolio Management](../Portfolio_Management_and_Optimization/)
- Sibling: [`../../Banking/Credit_Scoring_and_Underwriting/`](../../Banking/Credit_Scoring_and_Underwriting/) (credit risk)
- Fundamentals: [`../../../../01_AI_FUNDAMENTALS_AND_THEORY/Statistical_Learning/`](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Statistical_Learning/)

**Keywords:** fixed income, bonds, yield curve, duration convexity, Nelson-Siegel, credit spreads, rates models, bond risk premia machine learning, SOFR, MBS prepayment ML.
