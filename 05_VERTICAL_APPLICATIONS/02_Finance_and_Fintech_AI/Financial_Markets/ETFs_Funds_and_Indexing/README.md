# ETFs, Funds & Indexing

> ETFs and funds package diversified exposure into a single tradable instrument. This page covers index construction, passive vs. active, smart beta, robo-advisors, and ML for replication and allocation.

## Core concepts

- **Vehicles**: ETFs (exchange-traded), mutual funds, index funds, closed-end funds, UCITS; **creation/redemption** (authorized participants) keeps ETF price ≈ NAV.
- **Indexing**: market-cap vs. equal vs. fundamental weighting; float adjustment; rebalancing; tracking error; the index → fund replication problem (full replication vs. **sampling/optimization**).
- **Passive vs. active**: the low-cost passive wave (Vanguard/iShares/SPDR); active ETFs; the cost/After-fee performance debate.
- **Smart beta / factor ETFs**: rules-based tilts to value, momentum, quality, low-vol, size.
- **Direct indexing**: personalized, tax-loss-harvested portfolios at the security level.

## ML / AI applications

| Application | Idea |
|---|---|
| Index replication / sampling | Optimization & ML select a subset that tracks an index with min tracking error / cost |
| Robo-advisors | Automated allocation, rebalancing, tax-loss harvesting (Betterment, Wealthfront) — see [Portfolio Management](../Portfolio_Management_and_Optimization/) |
| Smart-beta factor timing | ML to time/blend factor exposures |
| Fund-flow & style analysis | Predict flows, classify style, detect "closet indexing" |
| Thematic/AI-driven ETFs | NLP/alt-data to build thematic baskets |
| Tax-loss harvesting | Optimization over lots & wash-sale constraints (direct indexing) |

## Data & tools

- ETF holdings/flows: issuer sites, ETF.com, `yfinance`, FactSet. Index methodologies: MSCI, S&P DJI, FTSE Russell. Replication/optimization: `cvxpy`, `PyPortfolioOpt`, `riskfolio-lib`.

## Key references

- Sharpe, *The Arithmetic of Active Management* (1991) — the passive case.
- Malkiel, *A Random Walk Down Wall Street* — indexing rationale.
- Arnott, Hsu, Moore, *Fundamental Indexation* (FAJ 2005).

## 🌎 ETF deep-dives by country
| Page | What's inside |
|---|---|
| [Brazilian ETFs on B3](./Brazilian_ETFs_B3.md) | BOVA11, IVVB11, SMAL11, BOVV11, PIBB11, DIVO11, GOLD11, fixed-income (IMAB11, IB5M11, FIXA11), thematic (NASD11, HASH11), issuers, costs, tributação de ETF. |
| [US ETFs](./US_ETFs.md) | SPY/IVV/VOO, QQQ, VTI, bond/intl ETFs, sector SPDRs, spot bitcoin ETFs (IBIT/FBTC), leveraged, issuers, creation/redemption, access from abroad. |

## Related in AIForge
- [Portfolio Management & Optimization](../Portfolio_Management_and_Optimization/) · [Equities & Stock Markets](../Equities_and_Stock_Markets/) · [Technical & Fundamental Analysis](../Technical_and_Fundamental_Analysis/)
- Fundamentals: [`../../../../01_AI_FUNDAMENTALS_AND_THEORY/Classical_ML_Algorithms/`](../../../../01_AI_FUNDAMENTALS_AND_THEORY/Classical_ML_Algorithms/)

**Keywords:** ETF, index funds, passive investing, smart beta, factor ETF, robo-advisor, index replication, direct indexing, tax-loss harvesting, fundamental indexation.
