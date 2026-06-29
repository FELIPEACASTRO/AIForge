# US ETFs (ETFs Americanos)

> Reference guide to US-listed exchange-traded funds (ETFs) — the ecosystem, the biggest funds by AUM, issuers, creation/redemption mechanics, fees, smart-beta and active strategies, spot bitcoin ETFs, how non-US (especially Brazilian) investors access them, and the ML/quant angle of index replication.

An **ETF (Exchange-Traded Fund / fundo negociado em bolsa)** is a pooled investment vehicle whose shares trade intraday on an exchange like a stock, while typically tracking an index, sector, commodity, or strategy. In the US, most ETFs are registered open-end investment companies under the Investment Company Act of 1940. Since **SEC Rule 6c-11** (adopted 25 Sep 2019, effective 23 Dec 2019), most fully transparent, index-based and active ETFs can launch without bespoke exemptive orders, standardizing the regulatory framework ([SEC press release 2019-190](https://www.sec.gov/newsroom/press-releases/2019-190), [SEC compliance guide](https://www.sec.gov/investment/exchange-traded-funds-small-entity-compliance-guide)).

The US ETF market is the world's largest and is still growing rapidly: US ETFs pulled in a **record ~$1.49 trillion of net inflows in 2025** (industry trackers report a range of ~$1.46T–$1.50T depending on methodology), pushing US ETF assets past **~$13 trillion** ([etf.com](https://www.etf.com/sections/monthly-etf-flows/us-etfs-pull-record-149-trillion-2025), [ETFGI](https://etfgi.com/news/press-releases/2026/01/etfgi-reports-record-us150-trillion-2025-net-inflows-push-us-etf)). SPY, launched **22 January 1993**, was the first US-listed ETF ([SSGA — SPY](https://www.ssga.com/us/en/intermediary/etfs/spdr-sp-500-etf-trust-spy)).

---

## Largest US ETFs by AUM (late 2025 / early 2026)

AUM figures are approximate (snapshot ~late 2025/early 2026) and move daily; expense ratios are the net ratios published by issuers. Verify the live figure on the issuer fund page before relying on it.

| Ticker | Fund | Issuer | Exposure | Expense ratio | Approx. AUM |
|--------|------|--------|----------|---------------|-------------|
| **VOO** | Vanguard S&P 500 ETF | Vanguard | S&P 500 | 0.03% | ~$1T+ |
| **SPY** | SPDR S&P 500 ETF Trust | State Street (SSGA) | S&P 500 | 0.0945% | ~$750B+ |
| **IVV** | iShares Core S&P 500 ETF | BlackRock iShares | S&P 500 | 0.03% | ~$690B+ |
| **VTI** | Vanguard Total Stock Market ETF | Vanguard | US total market | 0.03% | ~$545B |
| **QQQ** | Invesco QQQ Trust | Invesco | Nasdaq-100 | 0.18% | ~$380B |
| **VUG** | Vanguard Growth ETF | Vanguard | US large-cap growth | 0.04% | ~$194B |
| **VEA** | Vanguard FTSE Developed Markets ETF | Vanguard | Developed ex-US | ~0.03% | ~$180B |
| **IEFA** | iShares Core MSCI EAFE ETF | iShares | Developed ex-US/Canada | ~0.07% | ~$155B |

Sources: [etf.com — 10 biggest ETFs](https://www.etf.com/sections/features/10-biggest-etfs-us); [SSGA SPY page](https://www.ssga.com/us/en/intermediary/etfs/spdr-sp-500-etf-trust-spy). In 2025 **VOO overtook SPY** (Feb 2025) to become the world's largest ETF, drawing a record single-fund annual inflow (~$137B) and becoming the **first ETF ever to surpass $1 trillion in assets** ([etf.com — VOO becomes first $1T ETF](https://www.etf.com/sections/features/voo-becomes-first-1-trillion-etf), [etf.com — VOO record flows](https://www.etf.com/sections/features/voos-record-shattering-run-shows-no-signs-slowing)).

> **Note on SPY vs VOO/IVV:** SPY remains a **unit investment trust (UIT)**, which constrains securities lending and dividend reinvestment and explains its higher 0.0945% fee vs the 0.03% charged by VOO/IVV ([SSGA SPY](https://www.ssga.com/us/en/intermediary/etfs/spdr-sp-500-etf-trust-spy)). **QQQ** completed its conversion from a UIT to an open-end fund in **December 2025** (shareholders approved on 19 Dec 2025; QQQ began trading as an open-end fund on 22 Dec 2025), cutting its fee from 0.20% to **0.18%** and unlocking securities lending and income reinvestment ([Invesco — QQQ modernization](https://www.prnewswire.com/news-releases/invesco-qqq-shareholders-vote-to-approve-modernization-302646935.html), [Invesco — What's new about QQQ](https://www.invesco.com/qqq-etf/en/market-outlook/whats-new-about-qqq.html)).

---

## The big four issuers

| Issuer (brand) | Parent | Notable franchises |
|----------------|--------|--------------------|
| **iShares** | BlackRock | IVV, AGG, IEFA, EEM/EFA, factor/min-vol suite, **IBIT** (bitcoin) |
| **Vanguard** | Vanguard | VOO, VTI, VEA, VWO, BND, VUG/VTV |
| **SPDR** | State Street (SSGA) | SPY, GLD, the 11 **Select Sector SPDRs** (XLK, XLF, …) |
| **Invesco** | Invesco | QQQ, equal-weight RSP, factor PowerShares lineage |

Together with Fidelity (FBTC, low-cost core funds), these issuers dominate US ETF assets.

---

## Core building blocks by asset class

| Ticker | Fund | Asset class | Expense ratio |
|--------|------|-------------|---------------|
| **VOO / IVV / SPY** | S&P 500 trackers | US large-cap equity | 0.03% / 0.03% / 0.0945% |
| **VTI** | Total US stock market | US all-cap equity | 0.03% |
| **QQQ** | Nasdaq-100 | US large-cap growth/tech | 0.18% |
| **VEA / IEFA** | Developed ex-US | Intl developed equity | ~0.03% / ~0.07% |
| **VWO / EEM** | Emerging markets | EM equity | ~0.07% / ~0.70% |
| **AGG** | iShares Core US Aggregate Bond | US investment-grade bonds | 0.03% |
| **BND** | Vanguard Total Bond Market | US investment-grade bonds | low (≈0.03%) |
| **GLD** | SPDR Gold Shares | Gold (physical) | 0.40% |

Sources: [iShares AGG](https://www.ishares.com/us/products/239458/ishares-core-total-us-bond-market-etf); [SSGA GLD](https://www.ssga.com/us/en/intermediary/etfs/spdr-gold-shares-gld). **AGG** tracks the Bloomberg US Aggregate Bond Index; **GLD** holds allocated physical gold bullion (London Good Delivery bars) in vaults.

### Sector SPDRs (Select Sector, 0.08% each)

The 11 GICS-sector SPDRs slice the S&P 500. All share a **0.08%** net expense ratio ([SSGA XLK](https://www.ssga.com/us/en/intermediary/etfs/state-street-technology-select-sector-spdr-etf-xlk), [SSGA XLF](https://www.ssga.com/us/en/intermediary/etfs/state-street-financial-select-sector-spdr-etf-xlf)).

| Ticker | Sector | Ticker | Sector |
|--------|--------|--------|--------|
| XLK | Technology | XLP | Consumer Staples |
| XLF | Financials | XLB | Materials |
| XLE | Energy | XLU | Utilities |
| XLV | Health Care | XLRE | Real Estate |
| XLY | Consumer Discretionary | XLC | Communication Services |
| XLI | Industrials | | |

---

## Smart-beta / factor ETFs (smart beta / fatores)

Factor ETFs deviate from market-cap weighting to harvest documented risk premia: **value, size, momentum, quality, low/minimum volatility** ([iShares factor lineup](https://www.ishares.com/us/products/251614/ishares-msci-usa-momentum-factor-etf)).

| Ticker | Factor | Index family |
|--------|--------|--------------|
| **MTUM** | Momentum | MSCI USA Momentum (risk-adjusted 6- & 12-month price momentum, vol-scaled over 3 years) |
| **VLUE** | Value | MSCI USA Enhanced Value |
| **QUAL** | Quality | MSCI USA Quality |
| **USMV** | Minimum volatility | MSCI USA Minimum Volatility ([iShares USMV](https://www.ishares.com/us/products/239695/ishares-msci-usa-minimum-volatility-etf)) |
| **SIZE** | Size | MSCI USA Size factor |
| **RSP** | Equal weight | S&P 500 Equal Weight |

> **MTUM methodology detail:** MSCI builds a **risk-adjusted price momentum** score — excess return over the risk-free rate divided by the annualized standard deviation of weekly returns over the trailing ~3 years — computed over **6- and 12-month** windows, standardized, and combined into a single momentum score; the index then weights by momentum score × market-cap weight, with issuer caps, rebalancing semiannually ([MSCI Momentum methodology](https://www.msci.com/index/methodology/latest/Momentum)).

---

## Active ETFs (ETFs ativos)

Active ETFs are the fastest-growing segment. Global active-ETF assets hit a record **~$1.86 trillion by Nov 2025** ([ETFGI](https://etfgi.com/news/press-releases/2025/12/etfgi-reports-assets-invested-actively-managed-etfs-listed-globally)), with roughly a thousand new active ETFs launched in 2025 and active funds capturing a large and growing share of US ETF inflows ([InvestmentNews](https://www.investmentnews.com/etfs/record-setting-surge-in-active-etfs-redefines-2025-product-landscape/265052)). The archetype is **ARKK** (ARK Innovation ETF), Cathie Wood's actively managed disruptive-innovation fund ([ARK](https://www.ark-funds.com/funds/arkk)).

---

## Leveraged & inverse ETFs (alavancados/inversos) — trade with caution

| Ticker | Objective | Leverage | Expense ratio |
|--------|-----------|----------|---------------|
| **TQQQ** | ProShares UltraPro QQQ | +3× daily Nasdaq-100 | 0.82% |
| **SQQQ** | ProShares UltraPro Short QQQ | −3× daily Nasdaq-100 | 0.82% |

These target a **daily** multiple and **reset at each close**, so multi-day returns compound path-dependently. In choppy, sideways markets this causes **volatility decay (beta slippage)**: e.g., a −10% day followed by a +11.1% day (index roughly flat) leaves a +3× fund down meaningfully ([ProShares TQQQ](https://www.proshares.com/our-etfs/leveraged-and-inverse/tqqq)). They are trading tools, not buy-and-hold holdings.

---

## Spot bitcoin ETFs (ETFs de bitcoin à vista)

After Grayscale's court win, the SEC approved the first **US spot bitcoin ETPs on 10 January 2024** (approving 11 Rule 19b-4 applications); the first ~10 funds began trading **11 Jan 2024** ([SEC — Gensler statement](https://www.sec.gov/newsroom/speeches-statements/gensler-statement-spot-bitcoin-011023), [Congress.gov CRS](https://www.congress.gov/crs-product/IF12573)).

| Ticker | Fund | Issuer | Custody | Fee |
|--------|------|--------|---------|-----|
| **IBIT** | iShares Bitcoin Trust | BlackRock | Coinbase Custody | 0.25% |
| **FBTC** | Fidelity Wise Origin Bitcoin Fund | Fidelity | Fidelity Digital Assets (in-house) | 0.25% |
| **GBTC** | Grayscale Bitcoin Trust ETF | Grayscale | Coinbase | 1.5% (legacy) |

IBIT is the largest spot-BTC ETF by assets and volume. On **29 July 2025 the SEC approved in-kind creation/redemption** for spot BTC/ETH ETPs, improving their tax and arbitrage efficiency ([SEC press release 2025-101](https://www.sec.gov/newsroom/press-releases/2025-101-sec-permits-kind-creations-redemptions-crypto-etps), [CoinDesk](https://www.coindesk.com/markets/2025/07/29/sec-approves-in-kind-redemptions-for-all-spot-bitcoin-ethereum-etfs)).

---

## How ETFs actually work: creation / redemption & APs

ETF shares are minted and destroyed in the **primary market** by **Authorized Participants (APs)** — clearing-agency members with a written agreement with the fund (defined under Rule 6c-11) ([U.S. Bank](https://www.usbank.com/corporate-and-commercial-banking/insights/institutional/fund-administration/role-of-authorized-participants-etfs.html), [SEC 2019-190](https://www.sec.gov/newsroom/press-releases/2019-190)).

- **Creation:** an AP delivers a defined **basket** of underlying securities (+ a cash balancing amount) to the fund and receives a block of new shares — a **creation unit**, typically **25,000–200,000 shares** (most commonly 50,000) ([Schwab Asset Management](https://www.schwabassetmanagement.com/content/understanding-etf-creation-and-redemption-mechanism), [Investor.gov — creation unit](https://www.investor.gov/introduction-investing/investing-basics/glossary/creation-unit)).
- **Redemption:** the reverse — the AP returns a creation unit and receives the basket of securities.

This **in-kind arbitrage** keeps the ETF's market price anchored to its NAV (intrinsic value): if the ETF trades rich, APs create and sell; if cheap, they buy and redeem. It also delivers the structural **tax efficiency** of the ETF wrapper (in-kind redemptions purge low-basis lots without realizing capital gains). Rule 6c-11 requires ETFs to be exchange-listed, to publish daily portfolio holdings, and to transact in creation units with APs ([SSGA — how ETFs are created](https://www.ssga.com/us/en/intermediary/resources/education/how-etfs-are-created-and-redeemed)).

---

## How non-US investors access US ETFs (acesso de não-residentes)

1. **Direct via a US/global broker.** Many international brokers (e.g., Interactive Brokers-style platforms) let non-US persons buy US-listed ETFs directly. Caveats: **US dividend withholding tax (statutory 30%, treaty-reduced for some countries)** and, for non-resident aliens, potential **US estate-tax exposure** on US-situs assets above thresholds — a common reason non-US investors prefer Ireland-domiciled UCITS equivalents (e.g., CSPX vs IVV). Confirm your own treaty and tax situation.
2. **UCITS mirror funds** (Ireland/Luxembourg) replicate the same indices with non-US tax treatment — outside the scope of this US-listed page but the standard workaround.
3. **BDR route on B3 (Brazil).** Brazilians and other eligible investors can buy **BDRs de ETF (Brazilian Depositary Receipts on ETFs)** — Brazil-listed receipts backed by foreign ETF shares, settling in **reais (R$)** on B3, with no need to remit money abroad ([B3 — BDR on ETF (EN)](https://finservices.b3.com.br/en/bdr-on-etf), [B3 — BDRs de ETF (PT)](https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/renda-variavel/brazilian-depositary-receipts-bdrs-de-etf.htm)).
   - Flagship example: **IVVB11** — a B3-listed instrument (a Brazilian index fund holding ≥95% in the US-listed **IVV**) that tracks the **S&P 500** in reais ([B3 borainvestir IVVB11](https://borainvestir.b3.com.br/cotacoes/etfs/IVVB11/), [BlackRock Brasil — IVVB11](https://www.blackrock.com/br/products/251902/ishares-sp-500-fi-em-cotas-de-fundo-de-ndice-inv-no-exterior-fund)).
   - **BDRX** is B3's index of unsponsored BDRs traded in Brazil, weighted by free-float-adjusted market cap ([Investidor10 BDRX](https://investidor10.com.br/indices/bdrx/)).
   - Eligibility follows current **CVM** regulation; since 2020 BDRs are open to retail (varejo) investors, and **non-residents may also acquire them** if they meet registration requirements. Note: the underlying foreign assets are **not registered with the CVM/B3** ([B3 — BDR on ETF](https://finservices.b3.com.br/en/bdr-on-etf)).

---

## ML / quant angle: index replication & tracking-error optimization

ETFs are a natural laboratory for quantitative methods because the core engineering problem — **track an index cheaply** — is an optimization with real costs.

- **Replication approaches.** *Full replication* (hold every constituent) minimizes tracking error but maximizes turnover/cost; *stratified sampling* groups constituents by sector/size and samples; *optimized sampling* picks a subset to minimize tracking error ([review, arXiv 2601.03927](https://arxiv.org/abs/2601.03927)).
- **Cardinality-constrained (sparse) index tracking.** Limiting the portfolio to *k* assets cuts trading cost but is **combinatorial / NP-hard**. ML and optimization tackle this with LASSO/elastic-net sparse regression, **recursive feature elimination (RFE)** for constituent pre-selection, and mixed-integer programming ([feature-selection two-stage approach, ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1568494626006381); [asset pre-selection, arXiv 2503.18609](https://arxiv.org/abs/2503.18609)).
- **Deep-learning replicators.** Autoencoders / variational autoencoders learn low-dimensional latent factors to build sparse tracking baskets; **deep reinforcement learning** frames tracking as sequential rebalancing under transaction costs; differentiable cardinality constraints make the *k*-asset selection end-to-end trainable ([DCC, arXiv 2412.17175](https://arxiv.org/abs/2412.17175)).
- **Adjacent quant uses:** factor-ETF construction and back-testing, smart-beta signal research (momentum/quality/min-vol scoring), ETF-vs-NAV **arbitrage and premium/discount prediction**, basket optimization for APs, regime detection for leveraged-ETF decay modeling, and pairs/sector rotation across the SPDRs.

---

## Data & APIs (dados e APIs)

| Tool / source | Use | Notes |
|---------------|-----|-------|
| **yfinance** (Python) | Free historical prices for ETFs/stocks | Unofficial Yahoo endpoints; great for prototyping, not for production/commercial use ([docs](https://ranaroussi.github.io/yfinance/reference/index.html)) |
| **Polygon.io** | Real-time + historical US equities/ETF/FX/crypto | Official Python client `polygon-io/client-python`; paid tiers ([client](https://github.com/polygon-io/client-python)) |
| **Alpaca** | Commission-free trading + market data API | Brokerage API; real-time and historical US market data |
| **Alpha Vantage / Finnhub / IEX** | Free-tier quotes and fundamentals | Beginner-friendly REST APIs |
| **Issuer fund pages** | Authoritative holdings, NAV, expense ratio | iShares / Vanguard / SSGA / Invesco publish daily holdings (required under Rule 6c-11) |
| **SEC EDGAR (N-PORT, 497K)** | Regulatory filings, portfolios, prospectuses | Primary source for specs and holdings ([sec.gov](https://www.sec.gov)) |

For ground-truth specs (expense ratio, index, holdings, AUM) always prefer the **issuer fund page** and **EDGAR filings** over aggregators — aggregator figures can lag.

---

## Key takeaways

- The US ETF market is the deepest and cheapest in the world; **VOO/IVV/SPY** anchor most portfolios, with VOO now the largest fund globally and the **first ETF to cross $1 trillion in assets**.
- Fees compress relentlessly: core S&P 500 funds at **0.03%**, sector SPDRs at **0.08%**, QQQ cut to **0.18%** after its December 2025 open-end conversion.
- The **creation/redemption + AP** mechanism is what makes ETFs liquid, NAV-anchored, and tax-efficient.
- **Spot bitcoin ETFs (IBIT/FBTC, Jan 2024)** mainstreamed crypto access; **active and factor ETFs** are the growth frontier.
- Non-US investors can go direct (mind 30% dividend withholding and US estate tax), via **UCITS mirrors**, or via **BDRs on B3** (e.g., IVVB11) in reais.
- Quant relevance is concrete: **sparse, cardinality-constrained index tracking** is an active ML research area with autoencoder, RFE, and deep-RL methods.

**Sources:** [SEC 6c-11](https://www.sec.gov/newsroom/press-releases/2019-190) · [SEC compliance guide](https://www.sec.gov/investment/exchange-traded-funds-small-entity-compliance-guide) · [etf.com biggest ETFs](https://www.etf.com/sections/features/10-biggest-etfs-us) · [etf.com 2025 flows](https://www.etf.com/sections/monthly-etf-flows/us-etfs-pull-record-149-trillion-2025) · [ETFGI 2025 flows](https://etfgi.com/news/press-releases/2026/01/etfgi-reports-record-us150-trillion-2025-net-inflows-push-us-etf) · [etf.com VOO first $1T](https://www.etf.com/sections/features/voo-becomes-first-1-trillion-etf) · [SSGA SPY](https://www.ssga.com/us/en/intermediary/etfs/spdr-sp-500-etf-trust-spy) · [SSGA XLK](https://www.ssga.com/us/en/intermediary/etfs/state-street-technology-select-sector-spdr-etf-xlk) · [SSGA GLD](https://www.ssga.com/us/en/intermediary/etfs/spdr-gold-shares-gld) · [Invesco QQQ](https://www.invesco.com/qqq-etf/en/market-outlook/whats-new-about-qqq.html) · [iShares AGG](https://www.ishares.com/us/products/239458/ishares-core-total-us-bond-market-etf) · [iShares MTUM](https://www.ishares.com/us/products/251614/ishares-msci-usa-momentum-factor-etf) · [MSCI Momentum methodology](https://www.msci.com/index/methodology/latest/Momentum) · [iShares USMV](https://www.ishares.com/us/products/239695/ishares-msci-usa-minimum-volatility-etf) · [ProShares TQQQ](https://www.proshares.com/our-etfs/leveraged-and-inverse/tqqq) · [SEC spot BTC approval](https://www.sec.gov/newsroom/speeches-statements/gensler-statement-spot-bitcoin-011023) · [SEC in-kind 2025-101](https://www.sec.gov/newsroom/press-releases/2025-101-sec-permits-kind-creations-redemptions-crypto-etps) · [ETFGI active ETFs](https://etfgi.com/news/press-releases/2025/12/etfgi-reports-assets-invested-actively-managed-etfs-listed-globally) · [ARK ARKK](https://www.ark-funds.com/funds/arkk) · [B3 BDR on ETF (EN)](https://finservices.b3.com.br/en/bdr-on-etf) · [B3 BDRs de ETF (PT)](https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/renda-variavel/brazilian-depositary-receipts-bdrs-de-etf.htm) · [B3 IVVB11](https://borainvestir.b3.com.br/cotacoes/etfs/IVVB11/) · [arXiv 2412.17175](https://arxiv.org/abs/2412.17175) · [arXiv 2503.18609](https://arxiv.org/abs/2503.18609) · [arXiv 2601.03927](https://arxiv.org/abs/2601.03927) · [polygon client](https://github.com/polygon-io/client-python) · [yfinance docs](https://ranaroussi.github.io/yfinance/reference/index.html)

**Keywords:** US ETFs, ETFs americanos, SPY, VOO, IVV, QQQ, VTI, exchange-traded fund, fundo negociado em bolsa, BlackRock iShares, Vanguard, State Street SPDR, Invesco, creation redemption, authorized participant, participante autorizado, expense ratio, taxa de administração, tracking error, erro de rastreamento, smart beta, fatores, factor investing, active ETF, ETF ativo, leveraged ETF, TQQQ, SQQQ, ETF alavancado, spot bitcoin ETF, IBIT, FBTC, ETF de bitcoin, sector SPDR, XLK, XLF, AGG, BND, GLD, index replication, replicação de índice, sparse index tracking, BDR, BDR de ETF, IVVB11, B3, CVM, S&P 500, Nasdaq-100, SEC Rule 6c-11, quant finance, machine learning finance
