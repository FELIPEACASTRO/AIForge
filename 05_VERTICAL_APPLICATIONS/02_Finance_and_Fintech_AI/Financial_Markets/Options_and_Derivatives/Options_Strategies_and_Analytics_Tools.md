# Options Strategies & Analytics Tools

> A practitioner's catalog of listed-option **strategies** plus the **strategy builders, options-flow scanners, and dealer-positioning analytics** that traders actually use in 2024-2026 — what each tool does, free vs paid, Brazil (B3 / 🇧🇷) equivalents, and an honest read on what the GEX/flow hype can and cannot deliver.

This page is the *applied* companion to the contract-spec references already in this repo ([US_Options_and_Derivatives.md](./US_Options_and_Derivatives.md), [B3_Options_and_Derivatives_Brazil.md](./B3_Options_and_Derivatives_Brazil.md), and the [Options Market Prediction](./Options_Market_Prediction/) folder). Those cover *what the instruments are and how to model them*. This one covers *how to structure trades and which third-party tooling visualizes, screens, and surfaces them*. We do **not** re-explain the Greeks, IV, settlement, or 0DTE mechanics here — see the spec pages.

> Disclaimer: this is an educational index, not investment advice. Options can lose 100% of premium (and more, when undefined-risk). Prices and feature tiers change constantly — confirm on each vendor's page before subscribing.

---

## 1. The Strategy Catalog (Catálogo de Estratégias)

Every listed-option position is a combination of long/short calls, puts, and stock. The table groups the standard playbook by **risk class**. "Defined risk" (risco definido) means maximum loss is known and capped at trade entry; "undefined risk" (risco indefinido) means loss can be large or theoretically unbounded (naked short options) and requires margin (margem).

| Strategy (Estratégia) | Structure (Estrutura) | When used (Quando usar) | Risk class & max loss |
|---|---|---|---|
| **Covered call** (lançamento coberto) | Long 100 shares + short 1 OTM call | Neutral-to-mildly-bullish; income on a holding | Undefined downside (stock can fall); upside capped |
| **Cash-secured put** (venda de put coberta por caixa) | Short 1 OTM put, cash set aside to buy shares | Willing to own the stock lower; collect premium | "Defined-ish": loss = strike − premium if assigned to ~0 |
| **The Wheel** (a roda) | CSP → if assigned, sell covered calls → repeat | Systematic income on names you'd own | Same as CSP/covered call legs |
| **Bull call spread** (trava de alta com calls) | Long lower-strike call + short higher-strike call | Moderately bullish, want lower cost than long call | Defined = net debit paid |
| **Bear put spread** (trava de baixa com puts) | Long higher-strike put + short lower-strike put | Moderately bearish | Defined = net debit paid |
| **Bull put spread** (trava de alta com puts) | Short higher-strike put + long lower-strike put | Bullish/neutral; net credit | Defined = strike width − credit |
| **Bear call spread** (trava de baixa com calls) | Short lower-strike call + long higher-strike call | Bearish/neutral; net credit | Defined = strike width − credit |
| **Long straddle** (compra de straddle) | Long ATM call + long ATM put | Expect a big move, direction unknown (e.g. earnings) | Defined = total debit; needs large move to profit |
| **Long strangle** (compra de strangle) | Long OTM call + long OTM put | Same as straddle, cheaper, wider breakevens | Defined = total debit |
| **Short straddle/strangle** | Short ATM/OTM call + put | Expect range-bound, sell IV | **Undefined** both sides; large margin |
| **Iron condor** (condor de ferro) | Bull put spread + bear call spread (4 legs) | Range-bound, defined-risk premium selling | Defined = wider width − net credit |
| **Iron butterfly** (borboleta de ferro) | Short ATM straddle + long OTM wings | Strong pin near a price; higher credit, tighter range | Defined = wing width − net credit |
| **Calendar spread** (calendário) | Short near-dated + long far-dated, same strike | Neutral; profit from time decay/term-structure | Defined ≈ net debit |
| **Diagonal spread** (diagonal) | Calendar with different strikes | Directional + time/vol view | Defined ≈ net debit |
| **Ratio spread** (trava razão) | Unequal long/short counts (e.g. 1×2) | Directional w/ financing; skew plays | Often **undefined** on the extra short leg |
| **Collar** (colar / financiamento) | Long stock + protective put + short call | Protect a gain cheaply; cap upside | Defined downside (put floor) |
| **LEAPS** (opções de longo prazo) | Long call/put >1yr to expiry | Long-horizon directional, lower theta drag | Defined = premium (if long) |
| **Poor Man's Covered Call (PMCC)** | Long deep-ITM LEAPS call + short shorter-dated OTM call | Covered-call income with ~70-80% less capital | Defined = net debit; watch early assignment/rolls |

Notes worth internalizing: **credit spreads** (iron condor, vertical credit spreads) profit from time decay and falling IV but cap profit at the credit; **debit spreads** need direction. The PMCC is a *call diagonal* used as a stock-replacement covered call — the long LEAPS substitutes for the 100 shares, freeing capital but adding theta/IV exposure on the long leg ([Option Alpha — PMCC](https://optionalpha.com/learn/poor-mans-covered-call)). Most US brokers permit defined-risk diagonals (incl. PMCC) in IRAs because the long leg caps risk. 🇧🇷 On B3, the *trava de alta/baixa* (vertical spreads), *borboleta*, *condor*, *straddle/strangle* (often called *compra/venda de volatilidade*) and *financiamento* (covered call) all trade with the same payoff logic — terminology differs, mechanics don't.

Strategy reference libraries (free): [OIC — Options Strategies Quick Guide](https://www.optionseducation.org/the-options-strategies-quick-guide), [Fidelity options strategy guide](https://www.fidelity.com/learning-center/investment-products/options/options-strategy-guide/overview), [tastylive learn center](https://tastylive.freshdesk.com/support/solutions/articles/48001141882-beginner-options-trading-course).

---

## 2. Strategy Builders & Visualizers (Construtores e Visualizadores de Payoff)

These tools let you assemble multi-leg positions and instantly see the **payoff diagram** (curva de payoff), breakevens, max profit/loss, probability of profit (POP), and combined Greeks — before risking capital. This is the single highest-ROI category for a new options trader.

| Tool | What it does | Cost | Notes / link |
|---|---|---|---|
| **OptionStrat** | Browser P&L builder, 50+ preset strategies, optimizer (rank trades by target price/date), unusual-flow feed | Freemium (free 15-min-delayed builder; **Live Tools ~$39.99/mo**, **Live Flow ~$99.99/mo**) | Polished iOS/Android apps; OPRA data. [optionstrat.com](https://optionstrat.com/) · [Builder](https://optionstrat.com/build) |
| **OptionsProfitCalculator** | 20+ named strategies (basic, spreads, advanced) + custom 2–8-leg builder; max profit/loss, breakeven, IV & view-date sliders, price-by-date P&L table | **Free** (optional paid ad-free) | The classic free calc; shareable links. [optionsprofitcalculator.com](https://www.optionsprofitcalculator.com/) |
| **OIC Strategy resources** (OCC) | Interactive Options Strategies Quick Guide + courses on the OIC site | **Free** | [optionseducation.org](https://www.optionseducation.org/the-options-strategies-quick-guide) |
| **thinkorswim — Analyze tab** (Schwab) | Risk Profile graphs, Probability Analysis, thinkBack historical replay, Strategy Roller (auto-roll covered calls), probability cones | Free with brokerage | Best-in-class desktop analytics. [Schwab — Analyze/Risk Profile](https://www.schwab.com/learn/story/analyze-trades-using-risk-profile-on-thinkorswim) · [Analyze docs](https://toslc.thinkorswim.com/center/howToTos/thinkManual/Analyze) |
| **Power E*TRADE** (Morgan Stanley) | Spectral Analysis "what-if" map (gain/delta/gamma/theta over price×time), Earnings Move Analyzer, strategy builder/snapshot | Free with brokerage | [Power E*TRADE tools](https://us.etrade.com/what-we-offer/our-accounts/power-etrade) |
| **Market Chameleon** | Option screeners (ATM IV, spreads, volume vs 90d avg), **option-chain backtester**, earnings/IV history | Free starter; Total Access (paid, ~15-min delayed) | Deep research, not real-time. [marketchameleon.com](https://marketchameleon.com/) · [Option Screener](https://marketchameleon.com/Screeners/Options) |
| **Tradytics** | AI signals from option premiums, gamma/delta-exposure (GEX) dashboards, live flow, AI trade ideas | Free tier; Pro ~$69/mo ($15 for 15-day trial) | Dense UI, ramp-up needed. [tradytics.com](https://tradytics.com/) |
| **Opstra (Definedge Securities)** 🇮🇳 | Strategy builder, 27+ tools: payoff, Greeks, IV/skew, OI analysis, PCR (India-focused but globally usable) | Freemium (Free + PRO) | [opstra.definedge.com/strategy-builder](https://opstra.definedge.com/strategy-builder) |

🇧🇷 **Brazil builders/visualizers** — see §6; the closest B3-native equivalents are **opcoes.net.br** (simulador de estratégias + Black-Scholes) and **OpLab/TradeMap** (payoff + Greeks on B3 series).

---

## 3. Options-Flow & Dealer-Positioning Analytics (Fluxo de Opções e Posicionamento de Dealers)

This is the most hyped — and most contested — corner of retail options tooling. Two distinct things get sold here:

1. **Options flow** = a real-time tape of large/unusual prints (sweeps, blocks), pitched as "follow the smart money."
2. **Dealer positioning** = modeled estimates of net dealer **gamma/delta exposure** (GEX/DEX) inferred from open interest, used to guess where hedging flows may pin or accelerate price.

| Tool | Primary signal | Cost (approx, 2025-26) | Honest note |
|---|---|---|---|
| **SpotGamma** | GEX key levels, **HIRO** real-time hedging indicator, **TRACE** heatmap, twice-daily Founder's Notes, Equity Hub (3,500+ names), TAPE flow | **Essential** & **Alpha** tiers; Alpha ~$149+/mo (HIRO/TRACE/Vol Dashboard) | Popularized "gamma exposure" for retail; methodology cited by desks & financial press. [spotgamma.com](https://spotgamma.com/) · [GEX explainer](https://spotgamma.com/gamma-exposure-gex/) |
| **MenthorQ** | Net Gamma Exposure, key gamma levels, "Blind Spot" levels, Q-Score, TradingView/NinjaTrader/Sierra overlays; covers stocks + futures options (ES, NQ, etc.) | Paid membership; free daily report | Strong TradingView integration; QUIN AI screener. [menthorq.com](https://menthorq.com/) · [What is GEX](https://menthorq.com/guide/what-is-gex/) |
| **Unusual Whales** | Real-time options flow, dark-pool prints, congressional trades, screeners, **API** (+ MCP server) | Plans from ~$48/mo (retail); separate API tokens | Broad data + developer API. [unusualwhales.com](https://unusualwhales.com/) · [Pricing](https://unusualwhales.com/pricing) |
| **Cheddar Flow** | Real-time flow scanner, intermarket-sweep detection, dark-pool data; GEX paired on Pro plan | From ~$35/mo; Pro ~$99/mo (7-day trial) | Processes ~500k+ contracts/day, sub-second updates claimed; no native app. [cheddarflow.com](https://www.cheddarflow.com/) |
| **FlowAlgo** | Institutional order-flow tape (sweeps, blocks), dark-pool | Paid (~$99-149/mo); still operating (since 2017) | Older UI; many users migrate to UW/Cheddar. [flowalgo.com](https://flowalgo.com/) |
| **SqueezeMetrics** | **DIX** (Dark Index, short-vol/dark-pool proxy) + **GEX** index for SPX; history back to 2011 | Paid Data plan; **DIX published free** on the monitor | The original public GEX/DIX research. [squeezemetrics.com/monitor/dix](https://squeezemetrics.com/monitor/dix) · [Docs](https://squeezemetrics.com/monitor/docs) |
| **Tier1 Alpha** | Institutional-grade positioning research: daily GEX models, Implied Volatility Ranges, MBAD indicator, machine-driven market structure | Paid (distributed exclusively via Hedgeye) | Hedgeye distribution partnership. [tier1alpha.com](https://tier1alpha.com/) |
| **Convexitas** | Discretionary equity-derivative overlay / tail-liquidity strategies (asset manager, not a retail dashboard) | Institutional / managed accounts | Adjacent to the same positioning/convexity research community. [convexitas.com](https://www.convexitas.com/) |

Free/low-cost GEX charts also exist at [Barchart (SPY gamma exposure)](https://www.barchart.com/etfs-funds/quotes/SPY/gamma-exposure) and [FlashAlpha free GEX tool](https://flashalpha.com/tools/gamma-exposure) — useful sanity checks against paid feeds.

### Is flow / GEX actually predictive? (Vale a pena? Honest take)

The **mechanics are real**: dealers do delta-hedge, and when they are net-short gamma their hedging *amplifies* moves (buy strength, sell weakness), while net-long gamma *dampens* them. This is well documented and even appears in academic price-impact work. **But the predictive story is much thinner than the marketing.** A pre-registered FlashAlpha backtest on **1,972 SPY days (2018–2026)** found raw GEX correlates with next-day realized vol (ρ ≈ −0.36), but after controlling for VIX + ATM IV the signal collapses to ρ ≈ −0.03 (not significant) — i.e. GEX is best understood as a *structural/regime* descriptor, "modestly incremental over VIX," not a directional buy/sell signal. Several points of honest skepticism:

- Vendors compute GEX differently (open-interest sign assumptions, which expiries, dealer-long-vs-short modeling) — numbers diverge across SpotGamma, MenthorQ, SqueezeMetrics, Barchart for the *same* underlying.
- Daily OI snapshots miss intraday 0DTE dynamics that now dominate SPX hedging flows.
- For thin single-names, dealer flows are too small to systematically move price.
- "Unusual flow = smart money" is a weak inference: most large prints are hedges, spreads, or roll legs, not directional bets, and you only see one leg.

Treat these tools as **context, not signals**. They tell you whether you're in a stabilizing or destabilizing regime and where dealer hedging *may* cluster — not which way price goes. See [FlashAlpha — 8-year SPY GEX backtest (mostly tracks VIX)](https://flashalpha.com/articles/gex-dex-vex-chex-8-year-backtest-spy-vix-control) and the [SpotGamma GEX explainer](https://spotgamma.com/gamma-exposure-gex/).

---

## 4. Education & Research Desks (Educação e Mesas de Pesquisa)

| Source | What you get | Cost | Link |
|---|---|---|---|
| **tastylive / tastytrade research** | Hours/day of live market shows, free courses (bullish/bearish/neutral/advanced strategies), mechanics-heavy research (premium selling, POP, managing winners) | **Free** education (tastytrade = the broker) | [tastylive learn](https://tastylive.freshdesk.com/support/solutions/articles/48001141882-beginner-options-trading-course) · [tastytrade learn](https://tastytrade.com/learn/) · [courses](https://courses.tastytrade.com/) |
| **Option Alpha** | No-ads, structured options education + automation (bots) that connect to tastytrade | Free education; paid automation | [optionalpha.com](https://optionalpha.com/) · [PMCC guide](https://optionalpha.com/learn/poor-mans-covered-call) |
| **Cboe Options Institute** | 40+ yrs of free courses, tools, on-demand education from the exchange that lists SPX/VIX | **Free** | [cboe.com/optionsinstitute](https://www.cboe.com/optionsinstitute/) |
| **OIC — Options Industry Council** | Industry resource from OCC since 1992; free webinars, podcasts, courses, literature, strategy quick guide | **Free** | [optionseducation.org](https://www.optionseducation.org/) · [OCC Learning](https://www.optionseducation.org/theoptionseducationcenter/occ-learning) |

Why these matter: the exchange/clearing-funded desks (Cboe, OIC) are the **authoritative, conflict-light** sources for *mechanics and risk*; tastylive/Option Alpha are strong on *systematic premium-selling methodology* but carry a house style (sell IV, manage early, high-frequency small trades) you should read critically.

---

## 5. Quick Decision Guide (Por onde começar)

| Goal | Best free start | Paid upgrade if serious |
|---|---|---|
| Visualize a trade's payoff | OptionsProfitCalculator; broker Analyze tab | OptionStrat Live Tools |
| Screen for unusual activity | Barchart, Market Chameleon free tier | Unusual Whales / Cheddar Flow |
| Understand dealer regime (GEX) | FlashAlpha free GEX tool, SqueezeMetrics DIX | SpotGamma Alpha / MenthorQ |
| Learn mechanics properly | OIC + Cboe Options Institute (free) | — |
| Automate a strategy | Option Alpha (free education) | Option Alpha automation + tastytrade |
| 🇧🇷 Trade/analyze B3 options | opcoes.net.br (free) | OpLab / TradeMap Pro |

---

## 6. Brazil 🇧🇷 Tools (Ferramentas Brasileiras para Opções da B3)

The B3 retail-options scene has its own native toolset. None of the US flow/GEX vendors cover B3 series, so these are the practical local equivalents.

| Tool | What it does | Cost | Link |
|---|---|---|---|
| **opcoes.net.br** | Free strategy **simulador** (payoff + Greeks, time/vol scenarios), grade/matriz por strike-vencimento, 3D volatility view, and a **Black-Scholes calculator** for B3-listed series | Freemium (free registration; paid tiers) | [opcoes.net.br](https://opcoes.net.br/) · [Simulador](https://opcoes.net.br/simulador) |
| **OpLab** | Web platform for analysis/simulation of B3 options strategies, real-time B3 quotes, TradingView charts, 30+ preset strategies, portfolio mgmt, **API** (PRO) | ONE ~R$97/mo, PRO ~R$185/mo (cheaper annual) | [oplab.com.br](https://oplab.com.br/) · [Planos](https://oplab.com.br/planos/) |
| **TradeMap** | Broad B3 app (3M+ investors) with an **options module**: simulações, curvas de Smile e Volatilidade, Greeks, filtros, TradingView, broker connection | Free / paid tiers | [trademap.com.br](https://trademap.com.br/) |
| **Brokers' own option tools** | Clear (Grupo XP) and BTG Pactual offer in-house options/strategy screens and *boletas* with payoff/Greeks for clients | Free with brokerage account | Clear / BTG Pactual platforms |

Practical reality 🇧🇷: B3 single-name option liquidity concentrates in a handful of names (PETR4, VALE3, banks, BOVA11) plus Ibovespa index options, so screeners are less about "unusual flow" and more about finding *any* liquid series with a tight spread. opcoes.net.br + OpLab cover most of what a Brazilian retail/quant trader needs; the US dealer-positioning vendors have no B3 product.

---

**Sources:** [OptionStrat](https://optionstrat.com/), [OptionsProfitCalculator](https://www.optionsprofitcalculator.com/), [Market Chameleon](https://marketchameleon.com/), [Tradytics](https://tradytics.com/), [Opstra (Definedge)](https://opstra.definedge.com/), [Schwab thinkorswim Analyze](https://www.schwab.com/learn/story/analyze-trades-using-risk-profile-on-thinkorswim), [Power E*TRADE](https://us.etrade.com/what-we-offer/our-accounts/power-etrade), [SpotGamma](https://spotgamma.com/), [MenthorQ](https://menthorq.com/), [Unusual Whales](https://unusualwhales.com/), [Cheddar Flow](https://www.cheddarflow.com/), [FlowAlgo](https://flowalgo.com/), [SqueezeMetrics DIX/GEX](https://squeezemetrics.com/monitor/dix), [Tier1 Alpha](https://tier1alpha.com/), [Convexitas](https://www.convexitas.com/), [tastylive](https://tastytrade.com/learn/), [Option Alpha PMCC](https://optionalpha.com/learn/poor-mans-covered-call), [Cboe Options Institute](https://www.cboe.com/optionsinstitute/), [OIC / optionseducation.org](https://www.optionseducation.org/), [FlashAlpha GEX backtest](https://flashalpha.com/articles/gex-dex-vex-chex-8-year-backtest-spy-vix-control), [opcoes.net.br](https://opcoes.net.br/), [OpLab](https://oplab.com.br/), [TradeMap](https://trademap.com.br/).

**Keywords:** options strategies, covered call, cash-secured put, the wheel, vertical spread, iron condor, iron butterfly, straddle, strangle, calendar spread, diagonal, ratio spread, collar, LEAPS, poor man's covered call, PMCC, defined risk, undefined risk, payoff diagram, options flow, dealer positioning, gamma exposure, GEX, DEX, DIX, OptionStrat, OptionsProfitCalculator, SpotGamma, MenthorQ, Unusual Whales, Cheddar Flow, SqueezeMetrics, Tier1 Alpha, tastylive, Option Alpha, Cboe Options Institute, OIC; estratégias de opções, lançamento coberto, venda de put, trava de alta, trava de baixa, condor de ferro, borboleta, straddle, calendário, diagonal, colar, financiamento, risco definido, curva de payoff, fluxo de opções, posicionamento de dealers, exposição gama, opcoes.net.br, OpLab, TradeMap, B3.
