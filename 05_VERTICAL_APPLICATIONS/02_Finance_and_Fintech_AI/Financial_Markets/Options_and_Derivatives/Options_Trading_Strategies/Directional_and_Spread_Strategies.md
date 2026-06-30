# Directional & Spread Options Strategies

> Practical, risk-aware reference for bullish/bearish options structures — single legs, vertical spreads, backspreads, ratio spreads, risk reversals, synthetics, LEAPS, PMCC, collars, and stock replacement — with payoffs, breakevens, Greeks, and verified learning sources (global, 2024–2026). Audience: Brazil-heavy; English body with Portuguese terms and B3 notes.

**Not investment advice.** This is research/education only (*material educacional, não é recomendação de investimento*). Options can lose 100% of premium (buyers) or far more than the credit received (uncovered sellers). Confirm every mechanic with your broker and the OIC/CBOE before trading.

---

## 1. Mental model: direction × volatility × time

Every directional trade is a bet on **three axes**, not one:

| Axis | Greek | Buyer bias | Seller bias |
|---|---|---|---|
| Direction (*direção*) | Delta (Δ) | wants Δ to grow in their favor | wants underlying to stall/reverse |
| Volatility (*volatilidade*) | Vega (ν) | long premium → **long vega** (helped by rising IV) | short premium → **short vega** (helped by IV crush) |
| Time (*tempo*) | Theta (Θ) | pays decay → **negative theta** | collects decay → **positive theta** |
| Curvature | Gamma (Γ) | long options → **long gamma** | short options → **short gamma** (risk near strike) |

**Debit vs credit choice.** A *debit* structure (buy net) is long premium: positive gamma/vega, negative theta — you need the move to happen and/or IV to rise. A *credit* structure (sell net) is short premium: positive theta, negative gamma/vega — you profit from time passing and IV falling, but carry tail/assignment risk. tastylive's house style sells premium in high IV (IV Rank elevated), uses ~45 DTE, targets the ~16–30 delta short strike, and manages winners at **~50% of max profit** or near **21 DTE** ([tastytrade glossary: IVR](https://support.tastytrade.com/support/s/solutions/articles/43000567353)).

**No free money.** Defined-risk credit spreads cap loss but also cap profit and usually have loss > credit; undefined-risk legs (naked short, ratio short side, synthetic short) can blow through margin. IV crush after earnings, bid/ask slippage, commissions, and early assignment all erode edge.

---

## 2. Single-leg directional plays

| Strategy | Structure | Market view | Max profit | Max loss | Breakeven | Greeks profile | When to use |
|---|---|---|---|---|---|---|---|
| **Long Call** (*compra de call*) | Buy 1 call | Bullish, expects move soon | Unlimited | Premium paid (debit) | Strike + premium | +Δ +Γ +ν −Θ | Defined-risk leverage; want convexity when IV is low/fair |
| **Long Put** (*compra de put*) | Buy 1 put | Bearish | Strike − premium (large, not unlimited) | Premium paid | Strike − premium | −Δ +Γ +ν −Θ | Directional bearish or hedge; cheaper than shorting stock |
| **Long LEAPS Call** | Buy 1 long-dated (≥1y) ITM call | Bullish, patient | Unlimited | Premium paid | Strike + premium | +Δ (high if ITM) low Γ +ν, slow −Θ | Multi-month conviction with less daily theta drag |

Single longs are the cleanest expression of a view but suffer **theta bleed** and **IV crush** (buying rich IV before earnings is a classic trap). OIC strategy pages: [Long Put](https://www.optionseducation.org/strategies/all-strategies/long-put), [LEAPS Strategies](https://www.optionseducation.org/optionsoverview/leaps-strategies).

---

## 3. Vertical spreads (defined risk, two legs, same expiry)

Verticals reduce cost/theta of a single long by selling a further strike, capping both risk and reward. Pick by direction **and** debit-vs-credit (which depends on whether you want long or short premium).

| Strategy | Structure | View | Max profit | Max loss | Breakeven | Greeks | When |
|---|---|---|---|---|---|---|---|
| **Bull Call (debit)** (*trava de alta com calls*) | Buy lower call, sell higher call | Moderately bullish | Width − net debit | Net debit | Long strike + net debit | +Δ −Θ +ν (small) | Bullish, want defined risk, IV low/fair |
| **Bear Put (debit)** (*trava de baixa com puts*) | Buy higher put, sell lower put | Moderately bearish | Width − net debit | Net debit | Long strike − net debit | −Δ −Θ +ν (small) | Bearish, defined risk, IV low/fair |
| **Bull Put (credit)** (*trava de alta com puts*) | Sell higher put, buy lower put | Bullish/neutral | Net credit | Width − net credit | Short strike − net credit | +Δ +Θ −ν | Bullish + want theta; high IV |
| **Bear Call (credit)** (*trava de baixa com calls*) | Sell lower call, buy higher call | Bearish/neutral | Net credit | Width − net credit | Short strike + net credit | −Δ +Θ −ν | Bearish/range top; high IV |

Verified mechanics: OIC [Bull Call Spread](https://www.optionseducation.org/strategies/all-strategies/bull-call-spread-debit-call-spread), [Bull Put Spread](https://www.optionseducation.org/strategies/all-strategies/bull-put-spread-credit-put-spread), [Bear Call Spread](https://www.optionseducation.org/strategies/all-strategies/bear-call-spread-credit-call-spread). Note the **payoff equivalence**: a bull call and a bull put with the same strikes have the same risk graph; choose debit vs credit by IV regime (sell when IV high) and theta preference, not by "bullishness."

**Strike & delta targeting.** Credit-spread sellers commonly sell the **~30 delta** short strike and buy a further wing (e.g. ~10 delta), giving ~70% probability OTM but loss > credit (Option Alpha's tested SPY put-credit-spread bots use 0.30 short / 0.10 long; [Option Alpha SPY put credit spread backtest](https://optionalpha.com/blog/spy-put-credit-spread-backtest)). Debit-spread buyers often buy ~ATM/slightly ITM and sell at the target price.

**Exits.** Credit: take ~50% of max profit, or roll/close as short strike's delta roughly doubles (Option Alpha adjustment trigger). Debit: scale out into the move; don't hold to expiry hoping for full width (pin/assignment risk). Comparison: [Option Alpha — Credit vs Debit Spreads](https://optionalpha.com/learn/credit-spreads-vs-debit-spreads).

---

## 4. Backspreads & ratio spreads (asymmetric leg counts)

These break the 1:1 ratio. **Backspreads** buy more than they sell (net long convexity, want a big move). **Ratio spreads** sell more than they buy (net short premium, want a measured move — undefined risk on the extra short side).

| Strategy | Structure | View | Max profit | Max loss | Breakevens (approx) | Greeks | When |
|---|---|---|---|---|---|---|---|
| **Call Ratio Backspread** | Sell 1 lower call, buy 2 higher calls (often for credit) | Strongly bullish, big up-move | Unlimited (upside) | Limited: at long strike, ≈ width − net credit (loss if credit) | Lower: short strike + net credit; Upper: long strike + (width − net credit) | +Δ +Γ **+ν** Θ mixed | Expect violent upside; long vega; hedged on downside if entered for credit |
| **Put Ratio Backspread** | Sell 1 higher put, buy 2 lower puts | Strongly bearish, big down-move | Large (to zero) | Limited near long-put strike | Upper: short strike − net credit; Lower: long strike − (width − net credit) | −Δ +Γ +ν | Expect crash/gap down; long vega |
| **Call Ratio Spread (front)** | Buy 1 lower call, sell 2 higher calls | Mildly bullish to target, then capped | At short strike: width + net credit | **Unlimited above** (naked short leg) | Upside BE: short strike + max profit/contract | +Θ −ν −Γ above strike | Mild rally to a target with extra premium — *undefined upside risk* |
| **Put Ratio Spread (front)** | Buy 1 higher put, sell 2 lower puts | Mild drift down to target | At short strike: width + net credit | Large to zero below (extra short put) | Downside BE: short strike − max profit/contract | +Θ −ν | Mild decline to target; *undefined downside risk* |

Mechanics & breakeven formulas: [Options Playbook — Call Backspread](https://www.optionsplaybook.com/option-strategies/call-backspread), [TradeStation — Ratio Back Spreads](https://www.tradestation.com/learn/options-education-center/trading-big-market-moves-with-ratio-back-spreads/), [CFI — Call Ratio Back Spread](https://corporatefinanceinstitute.com/resources/derivatives/call-ratio-back-spread/). **Warning:** the extra *short* leg in a front-ratio spread is effectively naked — treat as undefined risk and size accordingly. Natenberg's *Option Volatility & Pricing* and McMillan's *Options as a Strategic Investment* both cover ratio/backspread Greeks in depth (see Sources).

---

## 5. Combos, synthetics & risk reversal

| Strategy | Structure | View | Max profit | Max loss | Breakeven | Greeks | When |
|---|---|---|---|---|---|---|---|
| **Risk Reversal / Combo (bullish)** (*combo*) | Sell OTM put, buy OTM call (same expiry) | Bullish, low-cost | Unlimited (upside) | Large (short put → put strike − net credit) | depends on net debit/credit between strikes | +Δ, vega/skew sensitive | Cheap bullish exposure; finances call with put credit — *undefined downside* |
| **Synthetic Long Stock** (*sintético comprado*) | Buy call + sell put, **same strike & expiry** | Bullish (replicate shares) | Unlimited | Strike − net debit (down to ~0) | Strike ± net premium | Δ≈+1.0, ν & Θ ≈ neutral (offset) | Replicate stock with less capital; arbitrage/financing plays |
| **Synthetic Short Stock** (*sintético vendido*) | Sell call + buy put, same strike & expiry | Bearish | Strike − net debit | Unlimited (short call) | Strike ± net premium | Δ≈−1.0 | Replicate a short; *undefined upside risk* |

OIC: [Synthetic Long Stock](https://www.optionseducation.org/strategies/all-strategies/synthetic-long-stock) (confirms IV/theta roughly offset, Δ≈ shares). Risk reversal guides: [Fidelity — Bullish split-strike synthetic / risk reversal](https://www.fidelity.com/learning-center/investment-products/options/options-strategy-guide/bullish-split-strike-synthetic), [Schwab — How the risk reversal works](https://www.schwab.com/learn/story/how-risk-reversal-options-strategy-works), [Option Alpha — Risk Reversal](https://optionalpha.com/strategies/reversal). The **same-strike** combo is a pure synthetic (put-call parity); a **split-strike** version (different strikes) is the directional risk reversal and reflects volatility skew.

---

## 6. Protective & stock-replacement structures (with shares)

| Strategy | Structure | View | Max profit | Max loss | Breakeven | Greeks | When |
|---|---|---|---|---|---|---|---|
| **Married Put / Protective Put** (*put protetora*) | Long 100 shares + buy 1 put | Bullish but want a floor | Unlimited (shares) | Limited: (entry − put strike) + put premium | Stock entry + put premium | +Δ (<1) +ν long put | Insurance on a long; earnings/event hedge |
| **Collar** (*collar / financiamento*) | Long shares + buy OTM put + sell OTM call | Mildly bullish, protect gains | Capped at call strike (+ net credit/− debit) | Limited to put strike floor | entry ± net premium | reduced Δ, low net ν | Lock in a gained position cheaply (put financed by call) |
| **Stock Replacement (deep-ITM call)** (*substituição com call ITM*) | Buy deep-ITM call (Δ≈0.8–0.9) instead of shares | Bullish, capital-efficient | Unlimited | Premium paid | Strike + premium | +Δ high, modest Γ/ν, some −Θ | Free up capital vs owning 100 shares; less downside than shares |

OIC: [Protective Put (Married Put)](https://www.optionseducation.org/strategies/all-strategies/protective-put-married-put), [Collar / Protective Collar](https://www.optionseducation.org/strategies/all-strategies/collar-protective-collar). A collar is structurally a long stock + risk reversal; a zero-cost collar sets strikes so call credit ≈ put debit.

---

## 7. Poor Man's Covered Call (PMCC) / long-call diagonal

A **diagonal** where a deep-ITM LEAPS call (Δ≈0.70–0.90, ≥12 months) replaces 100 shares, and you sell short-dated (~30–45 DTE) OTM calls (~0.30 delta) against it to harvest theta.

| Item | Detail |
|---|---|
| Structure | Buy long-dated deep-ITM call (long leg) + sell short-dated OTM call (short leg) |
| View | Neutral-to-mildly-bullish (*levemente altista*) |
| Capital | ~60–85% less than a real covered call (no 100 shares) |
| Max profit | ≈ (short strike − long strike) − net debit (when short call is ATM at its expiry) |
| Max loss | Net debit paid (if underlying collapses) |
| Greeks | net +Δ, +Θ from short leg, mixed ν, **−Γ near short strike** |
| Key rule | Long-leg strike width vs short-strike spread must leave room: pick long strike so short-call credit can't exceed the diagonal width |
| Risk | Short call can be assigned/go ITM faster than the LEAPS gains; ex-dividend early assignment on the short call |

Verified: [tastytrade — Long Call Diagonal Spread (PMCC)](https://tastytrade.com/learn/trading-products/options/long-call-diagonal-spread/), [Option Alpha — Poor Man's Covered Call](https://optionalpha.com/learn/poor-mans-covered-call). The short call repeatedly collected lowers effective cost basis on the LEAPS over time, but a sharp drop loses the debit and a sharp rally caps you at the short strike.

---

## 8. Selection cheat-sheet

| If you expect… | And IV is… | Consider |
|---|---|---|
| Big, fast up-move | low/fair | Long call, call backspread, stock replacement |
| Big, fast down-move | low/fair | Long put, put backspread |
| Moderate up | low | Bull call (debit) |
| Moderate up | high | Bull put (credit) |
| Moderate down | low | Bear put (debit) |
| Moderate down | high | Bear call (credit) |
| Slow grind up, income | any | PMCC / long-call diagonal |
| Protect a long you own | rising | Married put or collar |
| Replicate shares, less capital | low | Synthetic long or deep-ITM call |

**Liquidity & cost reality:** trade where bid/ask is tight; multi-leg spreads must be filled near the mid or edge evaporates. Wide markets, low open interest, and per-leg commissions punish ratio/backspread complexity most.

---

## 🇧🇷 B3 (Brazil) specifics

- **Underlyings & liquidity** (*liquidez*): the deepest equity-option books are concentrated in a handful of large-caps — **PETR4** (Petrobras), **VALE3** (Vale), **BBAS3**, **ITUB4** — plus the Ibovespa ETF **BOVA11**. Single-stock option volume on B3 is heavily dominated by PETR4 and VALE3; confirm current liquidity on a live chain ([opcoes.net.br](https://opcoes.net.br/), [OpLab](https://oplab.com.br/)) before sizing, since concentration shifts over time.
- **Style/exercise**: B3 equity options are historically **American-style calls** and (for many series) **European-style puts** — confirm per series, because early-exercise/assignment (*exercício antecipado*) changes management of credit spreads and PMCCs.
- **Series & tickers** (*séries*): B3 uses letter codes for month/right (e.g. A–L calls, M–X puts) appended to the underlying root; weekly options (*opções semanais*) now exist on the most liquid names ([B3 — Opções Semanais FAQ](https://www.b3.com.br/data/files/55/83/20/ED/CE98D8103152D4C8AC094EA8/B3%20Opcoes%20Semanais%20-%20Perguntas%20Frequentes.pdf)).
- **Tools** (*ferramentas*): [OpLab](https://oplab.com.br/) (analytics, Greeks, screeners) and [opcoes.net.br](https://opcoes.net.br/) (free chains, BE, distance-to-strike, e.g. [PETR4 chains](https://opcoes.net.br/opcoes/bovespa/PETR4)).
- **Liquidity caveat**: beyond the top names, B3 single-stock option books thin out fast — far-OTM and far-dated strikes can be untradeable. LEAPS-style long-dated series are scarce vs the US, so US-style PMCC/LEAPS plays are harder to run domestically.

---

## Learning & systematization sources (verified)

- **OIC / OptionsEducation.org** — free, exchange-backed strategy pages with payoff diagrams: [all-strategies index](https://www.optionseducation.org/strategies/all-strategies-en).
- **CBOE** — [Common Options Trading Strategies (PDF)](https://cdn.cboe.com/resources/options/Trading_Strategies.pdf).
- **Books** — Lawrence McMillan, *Options as a Strategic Investment*, 5th ed. ([Amazon, ISBN 9780735204652](https://www.amazon.com/Options-as-Strategic-Investment-Fifth/dp/0735204659)); Sheldon Natenberg, *Option Volatility & Pricing*, 2nd ed. ([Amazon, ISBN 9780071818773](https://www.amazon.com/Option-Volatility-Pricing-Strategies-Techniques/dp/0071818774)).
- **tastylive / tastytrade** — strategy mechanics & market studies: [Long Call Diagonal (PMCC)](https://tastytrade.com/learn/trading-products/options/long-call-diagonal-spread/), [IVR docs](https://support.tastytrade.com/support/s/solutions/articles/43000567353).
- **Option Alpha** — systematic spreads, backtests, automation: [credit vs debit spreads](https://optionalpha.com/learn/credit-spreads-vs-debit-spreads), [risk reversal](https://optionalpha.com/strategies/reversal).
- **The Options Playbook** — concise per-strategy mechanics: [Call Backspread](https://www.optionsplaybook.com/option-strategies/call-backspread).
- **Brokers/edu** — [Fidelity strategy guide](https://www.fidelity.com/learning-center/investment-products/options/options-strategy-guide/bullish-split-strike-synthetic), [Schwab learn](https://www.schwab.com/learn/story/how-risk-reversal-options-strategy-works), [TradeStation](https://www.tradestation.com/learn/options-education-center/trading-big-market-moves-with-ratio-back-spreads/).
- **🇧🇷 Brazil** — [OpLab](https://oplab.com.br/), [opcoes.net.br](https://opcoes.net.br/), [B3 weekly options FAQ](https://www.b3.com.br/data/files/55/83/20/ED/CE98D8103152D4C8AC094EA8/B3%20Opcoes%20Semanais%20-%20Perguntas%20Frequentes.pdf).

**Reminder:** education/research only — *não é recomendação de investimento*. Backtest, paper-trade, and account for commissions, slippage, assignment, and IV crush before risking capital.

**Keywords:** options strategies, opções, directional trading, vertical spread (trava), bull call spread (trava de alta), bear put spread (trava de baixa), credit spread (trava de crédito), debit spread (trava de débito), ratio backspread, ratio spread, risk reversal (combo), synthetic long/short (sintético), LEAPS, poor man's covered call (PMCC), diagonal spread, collar, married put (put protetora), protective put, stock replacement, deep-ITM call, Greeks (delta, gamma, theta, vega), IV crush (esmagamento de volatilidade), breakeven (ponto de equilíbrio), defined vs undefined risk (risco definido/indefinido), assignment (exercício/atribuição), B3, PETR4, VALE3, BOVA11, OpLab, opcoes.net.br, OIC, CBOE, tastylive, Option Alpha, McMillan, Natenberg
