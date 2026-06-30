# Exotic Options, Swaps & Structured Products

> Everything **beyond vanilla options and futures**: path-dependent exotics (barrier, Asian, lookback, autocallable), volatility derivatives (variance/vol swaps, VIX complex, dispersion), OTC swaps (IRS, CDS, TRS, OIS/SOFR, swaptions), and packaged structured products (autocallables, reverse convertibles, capital-protected notes, COE 🇧🇷). Payoffs, pricing methods (Monte Carlo / PDE / QuantLib), and verified open-source + market references. Free vs paid marked; Brazil (B3) equivalents noted.

This page complements — and does **not** repeat — the existing [Options & Derivatives README](./README.md), [US Options & Derivatives](./US_Options_and_Derivatives.md), [B3 Options & Derivatives (🇧🇷)](./B3_Options_and_Derivatives_Brazil.md), and [Options Market Prediction](./Options_Market_Prediction/). Those cover vanilla listed options, the Greeks, IV surfaces, microstructure, and ML prediction. Here the focus is **second/third-generation derivatives and OTC instruments**.

---

## 1. Exotic Options (Opções Exóticas)

"Exotic" = any option whose payoff is not the plain call/put. Most are **path-dependent** (the price path, not just the terminal spot, matters) and therefore have no closed form, so they are priced by **Monte Carlo**, **PDE/finite-difference**, or **lattice** methods. QuantLib (free, C++ with Python bindings via SWIG) is the reference open-source library; commercial desks use Murex, Numerix, FINCAD, or in-house C++.

| Exotic | Payoff / structure | Typical use | Pricing / reference |
|---|---|---|---|
| **Barrier** (knock-in / knock-out, single/double) | Vanilla payoff that activates (KI) or extinguishes (KO) if spot touches a barrier (continuous or discrete monitoring) | Cheaper directional bets; building block of autocallables | Analytic (Merton/Reiner-Rubinstein) for continuous; PDE/MC otherwise. [QuantLib BarrierOption / DoubleBarrierOption / PartialTimeBarrierOption](https://quantlib-python-docs.readthedocs.io/en/latest/instruments/options.html); FX desks use **Vanna-Volga** smile correction |
| **Asian** (average price/strike) | Payoff on the **arithmetic or geometric average** of spot over a window | Reduces manipulation & terminal-spot risk; common in commodities/FX | Geometric: closed form. Arithmetic: MC or [QuantLib `DiscreteAveragingAsianOption` / `ContinuousAveragingAsianOption`](https://quantlib-python-docs.readthedocs.io/en/latest/instruments/options.html) |
| **Lookback** (fixed/floating strike) | Pays off vs. the **max or min** spot over the life | "Buy the low / sell the high" with hindsight; expensive | PDE or MC; analytic for continuous-monitoring GBM |
| **Digital / Binary** (cash-or-nothing, asset-or-nothing) | Fixed payout if a condition is met at expiry; else zero | Event/range bets; coupon triggers in notes | Closed form (BS); but discontinuous payoff → unstable Greeks near strike |
| **Cliquet / Ratchet** | Series of forward-starting options; gains locked in ("ratcheted") per period, often with local/global caps & floors | Capital-protected equity participation | MC; sensitive to forward skew. (`CliquetOption` exists in QuantLib C++; it is **not** exposed in the QuantLib-Python instrument docs, so price via MC or the C++ engine) |
| **Chooser** | Holder later chooses call or put | View on volatility, not direction | Decomposes into call + put (put-call parity); analytic |
| **Compound** (option on an option) | Call/put on call/put | Staged investment / FX hedging of contingent exposure | Geske (1979) closed form; MC for complex |
| **Basket / Rainbow / Worst-of / Best-of** | Payoff on a portfolio or the worst/best of N assets | Multi-asset structured notes; correlation product | Multi-asset MC; correlation is the key risk |
| **Quanto** | Foreign underlying, payoff in domestic currency at fixed FX | Remove FX risk from foreign-equity bet | Drift adjustment for spot-vol-FX correlation |
| **Autocallable** (auto-redeemable) | Path-dependent note: pays a coupon and **auto-redeems early** if the underlying is above an autocall barrier on observation dates; capital at risk via a downside (often KI) barrier | Single most-sold retail structured payoff worldwide | MC (often **Heston / local-stochastic vol**, quasi-MC, Brownian-bridge for barrier accuracy); see [Markov-chain pricing, RDR 2024](https://link.springer.com/article/10.1007/s11147-024-09206-z) |

**Open-source pricing code (free):**
- [QuantLib](https://www.quantlib.org/) / [QuantLib-Python docs](https://quantlib-python-docs.readthedocs.io/) — barrier, double-barrier, partial-time barrier, Asian, basket, variance-swap engines (actively maintained; reference open-source pricer).
- [PROJ_Option_Pricing_Matlab (jkirkby3)](https://github.com/jkirkby3/PROJ_Option_Pricing_Matlab) — Barrier, Asian, Parisian, Lookback, Cliquet, Variance Swap, Swing, Forward-starting, Step, Fader via the PROJ/Fourier method. **Actively maintained** (last commit Nov 2024, ~200★, MATLAB).
- [GitHub topic: barrier-option](https://github.com/topics/barrier-option) and [option-pricing](https://github.com/topics/option-pricing) — many MC/PDE implementations (quality and maintenance vary; inspect before use).

---

## 2. Volatility & Variance Derivatives (Derivativos de Volatilidade)

Instruments whose payoff depends explicitly on **realized or implied volatility**, not price direction.

| Instrument | Payoff / structure | Use | Pricing / reference |
|---|---|---|---|
| **Variance swap** | Pays (realized variance − strike²) × notional; **replicable from a static strip of vanilla options** + delta hedge | Pure vol exposure; the canonical vol product | Demeterfi-Derman-Kamal / Carr-Lee replication; [Simple Variance Swaps, Martin (NBER w16884)](https://www.nber.org/papers/w16884); [QuantLib VarianceSwap (annotated source)](https://rkapl123.github.io/QLAnnotatedSource/d1/dfb/class_quant_lib_1_1_variance_swap.html) |
| **Volatility swap** | Pays (realized vol − strike) × notional (linear in vol, not variance) | Cleaner vol exposure but not statically replicable (convexity adjustment) | Approx. via variance swap + convexity correction |
| **Gamma swap** (weighted variance) | Variance swap weighted by spot/spot₀ each day | Reduces sensitivity to low-price crash regimes; used in dispersion | "Third-generation" vol product (Carr-Lee) |
| **VIX futures (VX) & Mini VIX** | Futures on the 30-day expected S&P 500 vol; Mini = 1/10 size | Hedge/trade equity vol, term structure | [Cboe VIX products](https://www.cboe.com/tradable-products/vix/), [VIX term structure](https://www.cboe.com/tradable-products/vix/term-structure/) |
| **VIX options** & **Options on VIX futures (UX)** | Options on the VIX index; UX = option-on-future launched **Oct 14, 2024** (accessible to futures-only participants) | Convex vol bets, tail hedges | [Cboe options on VIX futures hub](https://www.cboe.com/VIX-OOF-pipeline-hub/), [whitepaper PDF](https://cdn.cboe.com/resources/membership/Options_on_VX_Futures_Whitepaper.pdf) |
| **Dispersion / correlation swap** | Short index variance vs. long sum of constituent variances ⇒ short **realized correlation** | Dispersion trading desk strategy | [Jacquier & Slaoui, Variance Dispersion & Correlation Swaps (Imperial)](https://www.ma.imperial.ac.uk/~ajacquie/index_files/Jacquier,%20Slaoui%20-%20Dispersion.pdf) |
| **DSPX (Cboe S&P 500 Dispersion Index)** | Index of expected 30-day dispersion from SPX + single-stock option prices (launched Sep 27, 2023) | Benchmark/trade implied dispersion | [Cboe S&P 500 Dispersion Index (DSPX)](https://www.cboe.com/us/indices/dispersion/) |

🇧🇷 **Brazil:** the main equity-vol benchmarks are **IVol-BR** — an academic implied-volatility index from **NEFIN/USP** (proposed 2015, based on ~2-month Ibovespa option prices, VIX-style methodology) — and the official exchange index **S&P/B3 Ibovespa VIX (VXBR)**, published by B3 in partnership with S&P DJI. There is no liquid listed variance-swap market in Brazil; vol exposure is taken via options or OTC. See [B3 volatility indices](https://www.b3.com.br/pt_br/market-data-e-indices/indices/indices-em-parceria-s-p-dowjones/indices-de-volatilidade.htm) and the [B3 page](./B3_Options_and_Derivatives_Brazil.md).

---

## 3. Swaps & OTC Linear/Rate Derivatives (Swaps e Mercado de Balcão)

The largest derivatives market by notional. Post-LIBOR (USD LIBOR's last panel settings ceased **June 30, 2023**) the world moved to **risk-free rates (RFRs)**: SOFR (USD), €STR (EUR), SONIA (GBP), TONA (JPY). In Brazil the reference is the **DI (CDI)** / SELIC; the **swap pré × DI** is the workhorse.

| Swap | Structure | Use | Pricing / reference |
|---|---|---|---|
| **Interest-rate swap (IRS)** | Exchange fixed for floating coupons on a notional | Convert fixed↔floating exposure; rate hedging | Dual-curve (OIS discounting + projection); [QuantLib swaps](https://quantlib-python-docs.readthedocs.io/en/latest/instruments/swaps.html) |
| **Overnight index swap (OIS) / SOFR swap** | Fixed vs. compounded overnight RFR | Discounting curve; policy-rate expectations | [Building a SOFR curve in QuantLib-Python](https://medium.com/top-python-libraries/inside-quantlib-building-a-sofr-swap-curve-with-quantlib-python-820da9eaa20a); lookback/lockout days added in [QuantLib 1.35 (Jul 2024)](https://www.implementingquantlib.com/2024/06/different-swaps.html) |
| **Swaption** | Option to enter a swap at a future date | Hedge optionality in rate exposure; callable bonds | SABR (lognormal/normal/shifted) vol cube; [QuantLib swaptions](https://quantlib-python-docs.readthedocs.io/en/latest/instruments/swaptions.html) |
| **Credit default swap (CDS)** | Buyer pays spread; seller pays on a credit event | Hedge/trade default risk; basis trades | **ISDA Standard Model** is the market norm; [QuantLib `IsdaCdsEngine` (annotated source)](https://rkapl123.github.io/QLAnnotatedSource/d2/de0/class_quant_lib_1_1_credit_default_swap.html); wrapper: [CreditDefaultSwapPricer](https://github.com/bakera1/CreditDefaultSwapPricer) (C++; **stale** — last commit Aug 2023) |
| **Total-return swap (TRS)** | One leg pays the total return (price + income) of an asset; other pays a financing rate | Synthetic leverage / financing; off-balance-sheet exposure | Discount financing leg; model asset return |
| **Equity swap** | Exchange equity (or index) return for a funding rate | Gain index exposure without ownership; tax/regulatory | Variant of TRS |
| **Cross-currency swap (XCCY)** | Exchange principal + interest in two currencies | FX funding, basis trading | Requires XCCY basis curve; [FX-implied SOFR curve in QuantLib](https://medium.com/top-python-libraries/inside-quantlib-building-fx-implied-curve-referencing-usd-sofr-using-quantlib-python-part-i-1af65445b215) |

**Data & infrastructure (mostly free public dissemination):**
- [DTCC Public Price Dissemination Dashboard](https://pddata.dtcc.com/) — real-time CFTC **Part 43** swap transaction & pricing data (rates, credit, equity, FX). **Free**, near-real-time (with regulatory delays).
- [CFTC Real-Time Reporting](https://www.cftc.gov/LawRegulation/DoddFrankAct/Rulemakings/DF_18_RealTimeReporting/index.htm) — rules behind SDR dissemination; **UPIs** (Unique Product Identifiers from the DSB) required across credit/equity/FX/rates as of 2024.
- [ISDA](https://www.isda.org/) — master agreements, the **SIMM** (Standard Initial Margin Model), and the **CDM** below.

---

## 4. Structured Products (Produtos Estruturados)

Packaged combinations of a bond/deposit + embedded options/exotics, sold to retail/private-bank clients. Issuers file thousands of these — e.g. SEC Form **424B2** pricing supplements from [Morgan Stanley](https://www.sec.gov/Archives/edgar/data/895421/000183988224037878/ms4663_424b2-22560.htm) and [UBS](https://www.sec.gov/Archives/edgar/data/1114446/000183988224026224/ubs_424b2-15683.htm) are public, free, and a rich real-world dataset of payoff terms.

| Product | Structure | Risk/return profile | Pricing |
|---|---|---|---|
| **Autocallable note** | Zero-coupon bond + sold down-and-in put + digital coupons + autocall barrier | High coupon, capital at risk below KI barrier; early redemption | MC (Heston/LSV, quasi-MC); see §1 |
| **Reverse convertible** | Deposit + sold put | High fixed coupon; if underlying falls below strike you receive shares | Bond + short put |
| **Capital / principal-protected note (PPN)** | Zero-coupon bond + bought call (or exotic) | Principal returned at maturity + upside participation; opportunity cost | Bond + long option |
| **Barrier reverse convertible (BRC)** | Reverse convertible with a KI barrier | Coupon protected unless barrier breached | Bond + short down-and-in put |
| **Accumulator** ("I-kill-you-later") | Buy fixed quantity at a discount daily until a KO; doubled quantity below strike | Asymmetric — large losses in falling market | Strip of forwards + barriers |
| **Leveraged / participation notes** | Embedded leverage on index/basket | Amplified up/down | MC on basket |

🇧🇷 **Brazil — COE (Certificado de Operações Estruturadas):** the local structured-product wrapper, the direct analogue of a structured note. Two modalities under **CMN Resolution 4.263/2013** and **CVM Instruction 569/2015**: *Valor Nominal Protegido* (principal-protected) and *Valor Nominal em Risco* (principal at risk). Issued only by banks, **mandatorily registered at B3 (ex-CETIP)**, with a standardized **DIE** (Documento de Informações Essenciais). Refs: [B3 COE product page](https://www.b3.com.br/pt_br/produtos-e-servicos/registro/operacoes-estruturadas/certificado-de-operacoes-estruturadas-coe.htm), [B3 COE Operations Manual (PDF)](https://www.b3.com.br/data/files/AA/66/CD/34/EBC309105FE89209AC094EA8/Manual%20de%20Operacoes%20-%20COE.pdf), [ANBIMA COE](https://www.anbima.com.br/pt_br/informar/regulacao/informe-de-legislacao/certificados-de-operacoes-estruturadas-coe.htm).

---

## 5. Other Leverage & Listed Wrappers (Outros Instrumentos)

| Instrument | What it is | Where | Note |
|---|---|---|---|
| **Warrant** | Issuer-written long-dated option-like security on a stock/index/basket | EU/Asia retail; [Euronext structured products](https://live.euronext.com/en/products/structured-products) | Counterparty (issuer) risk; not exchange-cleared like listed options |
| **Turbo / Knock-out / Mini-future** | Leveraged certificate with a KO barrier = strike; touch ⇒ expires worthless | EU (UBS, Société Générale, etc.); [UBS turbo warrants](https://www.ubs.com/ch/en/services/guide/investments/articles/turbo-warrants.html), [Swissquote](https://www.swissquote.com/en-ch/private/inspire/blog/markets-instruments/turbo-warrants-high-leverage-trading-without-options) | Cheap leverage; total loss on KO |
| **CFD (Contract for Difference)** | Cash-settled bet on price change; margined, long/short | Banned for US retail; common EU/UK/AU/BR | High leverage; mostly negative retail outcomes (ESMA disclosures) |
| **Spread bet** | Tax-advantaged UK CFD variant | UK only | Same risk profile as CFD |
| **ETN (Exchange-Traded Note)** | Unsecured debt tracking an index (incl. vol, e.g. former VXX-type) | US/EU listed | **Issuer credit risk** (unlike ETFs); some vol ETNs have collapsed/closed |

🇧🇷 **Brazil:** no liquid local warrant/turbo market; leveraged retail exposure is via **mini contracts (WIN/WDO)**, options, and **COE** (above). CFDs are offered by some offshore brokers but are not B3-listed.

---

## 6. Core References & Datasets (Referências)

| Resource | Type | Free? | Why it matters |
|---|---|---|---|
| **Hull — *Options, Futures, and Other Derivatives*, 11th ed. (Pearson, ©2022)** | Textbook | Paid | The standard; 11th ed. covers OIS/RFR transition, rough vol, and ML in pricing/hedging. [Pearson](https://www.pearson.com/en-us/subject-catalog/p/options-futures-and-other-derivatives/P200000005938) |
| **[QuantLib](https://www.quantlib.org/) + [Python docs](https://quantlib-python-docs.readthedocs.io/)** | Library | Free (open-source) | Reference pricer for exotics, swaps, swaptions, CDS, variance swaps |
| **[Implementing QuantLib](https://www.implementingquantlib.com/) / [QuantLib Guide](https://www.quantlibguide.com/)** (L. Ballabio) | Book/blog | Free/paid | Authoritative how-to, incl. [different kinds of swaps](https://www.quantlibguide.com/Different%20kinds%20of%20swaps.html) |
| **[ISDA](https://www.isda.org/)** | Standards body | Free docs | Master agreements, SIMM margin model, CDS Standard Model, definitions |
| **[Common Domain Model (CDM)](https://cdm.finos.org)** | Open data/process model | Free (open-source, **hosted by FINOS**, Community Specification License 1.0) | Machine-readable representation of products, trades & lifecycle events; jointly stewarded by ISDA/ISLA/ICMA. Repo: [finos/common-domain-model](https://github.com/finos/common-domain-model); see also the [ISDA CDM infohub](https://www.isda.org/isda-solutions-infohub/cdm/) |
| **[DTCC PDD Dashboard](https://pddata.dtcc.com/)** | Swap transaction data | Free | Real-time public OTC swap prints (rates/credit/equity/FX) |
| **[Cboe VIX product suite](https://www.cboe.com/tradable-products/vix/)** | Market/data | Free pages, paid data | VIX/Mini VIX futures, VIX options, options on VIX futures, DSPX |

**Recent research (2021-2025, free arXiv/journals):**
- [Hedging and Pricing Structured Products Featuring Multiple Underlying Assets (arXiv 2411.01121)](https://arxiv.org/abs/2411.01121) — Sharma et al.; ML pricing (~250× faster than MC) and **Delta/Delta-Gamma/RL hedging** of autocallable baskets (ACM ICAIF 2024 workshop).
- [Pricing and hedging autocallable products by Markov chain approximation — Cui, Li & Zhang, *Review of Derivatives Research* 27 (2024)](https://link.springer.com/article/10.1007/s11147-024-09206-z).
- [Deep Learning Option Pricing with Market Implied Volatility Surfaces — Ding, Lu & Cheung (arXiv 2509.05911)](https://arxiv.org/abs/2509.05911) — VAE-compressed IV surface → fast NN pricing of American/Asian exotics.
- [Deep Learning for Exotic Option Valuation — Cao, Chen, Hull & Poulos (arXiv 2103.12551)](https://arxiv.org/abs/2103.12551) — NN surrogates (volatility-feature approach) for path-dependent exotics.
- [A Risk-Neutral Neural Operator for Arbitrage-Free SPX-VIX Term Structures — Zhang (arXiv 2511.06451)](https://arxiv.org/abs/2511.06451) — "ARBITER" operator-learning model, joint SPX/VIX, no-arbitrage constraints.
- [AI4Contracts: LLM & RAG-Powered Encoding of Financial Derivative Contracts — Mridul, Sloyan, Gupta & Seneviratne (arXiv 2506.01063)](https://arxiv.org/abs/2506.01063) — mapping legal swap terms into machine-readable form (CDM-adjacent).

---

## Limitations & Honest Notes

- **No closed forms.** Almost everything here needs MC/PDE; arithmetic-Asian, autocallable, and worst-of baskets are MC-only in practice. Greeks of digitals/barriers are unstable near the barrier — handle with smoothing.
- **Model risk dominates.** Autocallable and dispersion P&L depend critically on the **vol surface, skew, and correlation** assumptions (Heston/LSV/rough vol) — not just spot. Garbage calibration ⇒ garbage price.
- **Open-source maturity varies.** QuantLib and PROJ_Option_Pricing_Matlab are actively maintained; smaller wrappers (e.g. CreditDefaultSwapPricer, last updated Aug 2023) are useful as references but unmaintained — validate before production use.
- **Retail structured products carry hidden costs.** Issuer credit risk (ETNs, COE, warrants), wide bid-ask, and embedded fees often make payoffs unfavorable; SEC 424B2 / B3 DIE filings disclose terms — read them.
- **Data is the bottleneck.** Free public OTC data (DTCC PDD) is delayed and anonymized; granular swap/exotic pricing data is paid (ICE, Bloomberg, LSEG/Refinitiv, Numerix). Brazil OTC/COE data is fragmented across B3 registration and bank DIEs.
- **Currency/locale.** Brazil lacks liquid listed variance swaps, warrants, and turbos; the practical exotic/structured wrapper is the **COE**, plus OTC swaps (pré × DI) and mini contracts.

---

**Sources:** [QuantLib](https://www.quantlib.org/) · [QuantLib-Python docs](https://quantlib-python-docs.readthedocs.io/) · [ISDA](https://www.isda.org/) · [FINOS CDM](https://cdm.finos.org) · [finos/common-domain-model](https://github.com/finos/common-domain-model) · [DTCC PDD](https://pddata.dtcc.com/) · [CFTC Real-Time Reporting](https://www.cftc.gov/LawRegulation/DoddFrankAct/Rulemakings/DF_18_RealTimeReporting/index.htm) · [Cboe VIX](https://www.cboe.com/tradable-products/vix/) · [Cboe options on VIX futures](https://www.cboe.com/VIX-OOF-pipeline-hub/) · [Cboe DSPX](https://www.cboe.com/us/indices/dispersion/) · [B3 COE](https://www.b3.com.br/pt_br/produtos-e-servicos/registro/operacoes-estruturadas/certificado-de-operacoes-estruturadas-coe.htm) · [B3 volatility indices](https://www.b3.com.br/pt_br/market-data-e-indices/indices/indices-em-parceria-s-p-dowjones/indices-de-volatilidade.htm) · [ANBIMA COE](https://www.anbima.com.br/pt_br/informar/regulacao/informe-de-legislacao/certificados-de-operacoes-estruturadas-coe.htm) · [Hull 11e (Pearson)](https://www.pearson.com/en-us/subject-catalog/p/options-futures-and-other-derivatives/P200000005938) · [Implementing QuantLib](https://www.implementingquantlib.com/) · [PROJ_Option_Pricing_Matlab](https://github.com/jkirkby3/PROJ_Option_Pricing_Matlab) · [CreditDefaultSwapPricer](https://github.com/bakera1/CreditDefaultSwapPricer) · [Martin, Simple Variance Swaps (NBER w16884)](https://www.nber.org/papers/w16884) · [Jacquier & Slaoui, Dispersion](https://www.ma.imperial.ac.uk/~ajacquie/index_files/Jacquier,%20Slaoui%20-%20Dispersion.pdf) · [Autocallable ML hedging (arXiv 2411.01121)](https://arxiv.org/abs/2411.01121) · [Autocallable Markov-chain pricing (RDR 2024)](https://link.springer.com/article/10.1007/s11147-024-09206-z) · [DL Option Pricing w/ IV Surfaces (arXiv 2509.05911)](https://arxiv.org/abs/2509.05911) · [DL for Exotic Option Valuation (arXiv 2103.12551)](https://arxiv.org/abs/2103.12551) · [SPX-VIX Neural Operator (arXiv 2511.06451)](https://arxiv.org/abs/2511.06451) · [AI4Contracts (arXiv 2506.01063)](https://arxiv.org/abs/2506.01063) · [SOFR curve in QuantLib](https://medium.com/top-python-libraries/inside-quantlib-building-a-sofr-swap-curve-with-quantlib-python-820da9eaa20a) · [Euronext structured products](https://live.euronext.com/en/products/structured-products) · [UBS turbo warrants](https://www.ubs.com/ch/en/services/guide/investments/articles/turbo-warrants.html)

**Keywords:** exotic options, barrier option, knock-in, knock-out, Asian option, lookback option, digital binary option, cliquet ratchet, chooser, compound option, basket worst-of, quanto, autocallable, variance swap, volatility swap, gamma swap, dispersion trading, correlation swap, VIX futures, VIX options, options on VIX futures, DSPX, interest-rate swap, IRS, OIS, SOFR, swaption, credit default swap, CDS, total-return swap, TRS, equity swap, cross-currency swap, structured products, reverse convertible, principal-protected note, accumulator, warrant, turbo, knock-out certificate, CFD, ETN, ISDA, CDM, SIMM, DTCC, QuantLib, Monte Carlo, PDE, deep hedging — opções exóticas, opção com barreira, opção asiática, opção lookback, autocallable, swap de variância, swap de volatilidade, negociação de dispersão, swap de taxa de juros, swap de crédito (CDS), swaption, produtos estruturados, COE, Certificado de Operações Estruturadas, debênture conversível reversa, nota de capital protegido, warrant, turbo, CFD, ETN, precificação Monte Carlo, derivativos de balcão (OTC), B3, CETIP, ANBIMA.
