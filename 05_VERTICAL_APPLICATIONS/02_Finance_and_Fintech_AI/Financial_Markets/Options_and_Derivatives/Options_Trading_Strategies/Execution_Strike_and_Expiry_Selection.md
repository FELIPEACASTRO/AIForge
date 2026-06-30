# Micro: Execution, Strike & Expiry Selection

> The *micro* layer of options trading: once you've chosen a strategy, **how** do you pick the exact strike, the exact expiration, and **how** do you actually get filled without bleeding edge to slippage, assignment, and pin risk? Payoff/breakeven/Greeks math, order mechanics, liquidity screening, and Brazil/B3 execution realities. Education/research only — **not investment advice**.

This page deliberately goes one level *below* the sibling strategy pages (`Strategy_Selection_and_Playbooks.md`, `Volatility_Income_and_Neutral_Strategies.md`). It assumes you've already decided *what* structure to trade and answers the next three questions: **which strike, which expiry, and how to execute.**

---

## 1. The micro decision stack

| Step | Question | Primary tool | Output |
|---|---|---|---|
| 1 | How far can the underlying move? | **Expected move** (EM) from IV | A price range / σ band |
| 2 | Which strike? | **Delta ≈ prob. ITM**, EM band, skew | Specific strike(s) |
| 3 | Which expiry? | **Theta vs gamma** tradeoff by DTE | A DTE bucket (e.g. ~45) |
| 4 | Is it tradable? | **Liquidity screen** (spread, OI, volume) | Go / no-go |
| 5 | How to get filled? | **Order type & routing** | Limit at/near mid, multi-leg as one order |
| 6 | What at expiry? | **Pin / assignment / exercise** rules | Close-or-hold plan |

> The order matters: strike and expiry are **downstream of the expected move**, and *all* of it is downstream of whether the contract is even liquid enough to enter and exit at a fair price.

---

## 2. Strike selection — expected move, delta, and skew

### Expected move (EM)

The 1-standard-deviation expected move over the life of the option:

```
EM ≈ S × IV × √(T / 365)
```

where `S` = underlying price, `IV` = annualized implied volatility (decimal), `T` = calendar days to expiry. Example: `S = 100`, `IV = 0.30`, `T = 30` → `EM ≈ 100 × 0.30 × √(30/365) ≈ 8.6`, i.e. a ~±$8.6 one-σ band (≈ $91.4–$108.6). Underliers stay inside the 1σ band ~68% of the time in theory ([projectoption](https://projectoption.com/expected-move-calculator); [OIC — Rule of 16](https://www.optionseducation.org/news/understanding-the-rule-of-16-in-plain-terms)).

**Rule of 16 (daily shorthand):** because √252 (trading days/yr) ≈ 15.87 ≈ 16, dividing annual IV by 16 approximates the **daily** expected move in percent. A stock at 32% IV implies roughly a ±2%/day move (`32 / 16 = 2`). Note this trading-day shortcut (√252) and the calendar-day EM formula above (√365) are two different conventions — don't mix them in one calculation ([OIC — Rule of 16](https://www.optionseducation.org/news/understanding-the-rule-of-16-in-plain-terms)).

### Delta as an approximate probability

| Short strike delta | Approx. prob. ITM at expiry | Typical use |
|---|---|---|
| **~50Δ** (ATM) | ~50% | Straddle body, max gamma/theta |
| **~30Δ** | ~30% | Aggressive credit-spread short strike |
| **~16Δ** | ~16% (≈ 1σ OTM) | Classic short-strangle / high-POP short strike |

Delta is a **rough** proxy for the risk-neutral probability of finishing in-the-money, not an exact one ([Schwab — delta & probability](https://www.schwab.com/learn/story/options-delta-probability-and-other-risk-analytics); [Option Alpha — delta for probabilities](https://optionalpha.com/lessons/using-delta-for-probabilities)).

> ⚠ **Skew caveat:** equity index puts trade at **higher IV** than equidistant calls (volatility skew/smile), so a 16Δ put sits *further* from spot than a 16Δ call, and delta-implied "probability" already bakes in that skew — it is **not** the real-world probability. Treat these as planning anchors, not guarantees ([Schwab](https://www.schwab.com/learn/story/options-delta-probability-and-other-risk-analytics)).

### Breakevens (memorize)

| Position | Breakeven |
|---|---|
| Long call | `strike + debit` |
| Long put | `strike − debit` |
| Short put / cash-secured put | `strike − credit` |
| Covered call | `cost basis − credit` |
| Credit spread | `short strike ∓ credit` |
| Long straddle | `strike ± total debit` |
| Iron condor | `short put − credit` and `short call + credit` |

---

## 3. Expiry selection — the theta / gamma tradeoff

| DTE bucket | Theta (decay) | Gamma (risk accel.) | Practical read |
|---|---|---|---|
| **0–7 DTE** | Maximal per day | **Explosive near ATM** | Day-trading only; tiny size; hard stops |
| **~21 DTE** | High | Rising fast | tastytrade "manage-by" line |
| **~30–45 DTE** | Strong, smoother | Manageable | Common premium-selling sweet spot |
| **60–90+ DTE** | Slow daily bleed | Low | Buyers waiting on a move; LEAPS-ish |

For a short option held into expiry, **theta and gamma rise together** as DTE shrinks — you collect decay faster but a small adverse move whips delta violently. This is why short-premium traders cut risk before the final stretch.

**tastytrade research defaults** (from large backtests, *not* laws): enter premium-selling spreads around **~45 DTE**, take profit at **~50% of max credit**, and **manage/close/roll at ~21 DTE** (whichever comes first) to dodge end-of-life gamma. Their large credit-spread study found 45-DTE-in / 21-DTE-managed produced the best risk-adjusted results vs 30 or 60 DTE ([Trader Central summary](https://traderc.com/21-dte-50-percent-profit-exit-options/)).

> These are regime-dependent heuristics. Backtested edges compress and reverse; size for the loss, not the win rate.

---

## 4. Order mechanics & execution

- **Use limit orders, not market orders.** Option spreads are wide; a market order pays the whole bid/ask. Start at or near the **mid**, then work the price.
- **Route multi-leg spreads as a single combo order** at a net debit/credit — never leg in manually unless you accept fill risk on the open leg.
- **Mid-price fills are not guaranteed.** On wide chains, expect to give up part of the spread.
- **Mind the close.** Liquidity and spreads worsen near the open and into expiry; pin-prone names get erratic late on expiration day.

---

## 5. Liquidity screening (go / no-go)

Trade only contracts that pass a basic liquidity screen — wide spreads silently destroy expected value:

| Check | Healthy sign |
|---|---|
| **Bid/ask spread** | Tight in absolute and % terms (e.g. a few cents on liquid names) |
| **Open interest (OI)** | Deep enough to exit, not just enter |
| **Volume** | Active same-day trading in the series |
| **Underlying** | Liquid name (SPX/SPY/QQQ; in 🇧🇷 PETR4/VALE3/BOVA11) |

If the spread is wide and OI thin, **the trade fails before the thesis is even tested.**

---

## 6. Slippage & transaction costs

- Commissions + slippage compound across **multi-leg** structures and frequent **rolling** — fatal to thin-credit, high-POP trades (a 90%-POP spread risking $9 to make $1 has almost no margin for fees).
- Always fold costs into the **expected value**: `EV = POP × avg_win − (1−POP) × avg_loss − costs`.
- Wider spreads = larger hidden cost on *both* entry and exit; count the round trip.

---

## 7. Expiration, pin risk & assignment

- **OCC auto-exercise:** US listed options that finish **$0.01 or more in-the-money** are automatically exercised under the "exercise-by-exception" procedure (OCC Rule 805); holders have until **5:30 p.m. ET** on expiration day to submit contrary (do-not-exercise / exercise) instructions ([OIC — Options Exercise FAQ](https://www.optionseducation.org/referencelibrary/faq/options-exercise); [FINRA Information Notice 02/03/21](https://www.finra.org/rules-guidance/notices/information-notice-020321)).
- **Pin risk:** when the underlying closes *right at* a short strike, you don't know if you'll be assigned — close before expiry to avoid waking up to an unexpected (and possibly partial) stock position.
- **Early exercise (American style):** short ITM calls are most at risk **just before ex-dividend**; deep-ITM puts can be exercised early too. Index options (SPX) are **European** — no early exercise.

### SPX/SPXW vs SPY (tax + settlement)

| | **SPX / SPXW (index)** | **SPY (ETF)** |
|---|---|---|
| Style | **European** | American |
| Settlement | **Cash-settled** (no shares) | Physical (shares) |
| Early-exercise risk | None | Yes |
| Tax (US) | **Section 1256: 60% long / 40% short-term**, regardless of holding period | Equity-option rules (ordinary short-/long-term) |

The IRS classifies broad-based index options as Section 1256 non-equity contracts; SPY options are equity options because SPY is a security/ETF ([Cboe — Index Options Tax Treatment](https://www.cboe.com/tradable_products/index-options-benefits-tax-treatment/)). *Tax rules change and depend on jurisdiction — confirm current treatment with a professional.*

---

## 8. 0DTE gamma tactics (high reward, high gamma)

Zero-days-to-expiration SPX options reached roughly **59% of full-year 2025 SPX option volume**, with a single-month record of **62.4% in August 2025** ([Cboe — SPX 0DTE record 62.4% share](https://www.cboe.com/insights/posts/spx-0-dte-options-jump-to-record-62-share-in-august/); [Cboe — State of the Options Industry 2025](https://www.cboe.com/insights/posts/the-state-of-the-options-industry-2025)).

- **Why people trade it:** all the decay happens *today*; defined-risk spreads cap loss; lots of POP-friendly setups.
- **Why it bites:** gamma is enormous near the money at 0DTE — a ~1% index move can swing a short strike's delta from ~0.05 to ~0.95 in hours, turning a "safe" condor into max loss with **no time to recover** and no IV mean-reversion to help.

> 0DTE is **active day-trading, not passive income.** Strictly defined-risk structures, hard stops, tiny size. Win-rate optics hide fat-tailed loss days.

---

## 9. Brazil / B3 execution specifics 🇧🇷

- **Style/settlement:** **opções sobre ações** (PETR4, VALE3, BBAS3, BOVA11) are **American-style** — early *exercício/atribuição* is possible, especially around dividends/JCP. **Opções sobre o Ibovespa (IBOV) are European-style and cash-settled.** Confirm specs per series.
- **Automatic exercise:** B3 auto-exercises equity/units/ETF options that finish **at least R$0,01 in-the-money** versus the underlying's closing price on expiration day; everything else expires worthless ([B3 — exercício automático](https://www.b3.com.br/pt_br/noticias/exercicio-de-opcoes-passa-a-ser-automatico-a-partir-de-maio.htm)).
- **Expiration & settlement:** monthly equity-option series expire on the **third Friday**; after exercise, share delivery settles in **D+2** ([B3 Educação — Opções](https://edu.b3.com.br/w/opcoes); [B3 Educação — Opções Semanais](https://edu.b3.com.br/w/opcoes-semanais)).
- **Liquidity reality:** depth concentrates in a handful of names and near-month series; **weeklies (semanais)** exist but are far thinner than US — favor the most-traded series and budget for wider spreads.
- **Tax (tributação):** IR on options/day-trade differs from spot and changes — consult current B3/Receita rules.
- **Tools:** [opcoes.net.br](https://opcoes.net.br/) (chains, IV history, Black-Scholes calculator, payoff), [OpLab](https://oplab.com.br/) (IV history, strategy/payoff builder, lançamento-coberto screens), [B3 Educação — Opções](https://edu.b3.com.br/w/opcoes).

---

## 10. Macro tie-in

Strike and expiry choices interact with the **macro calendar**: known binaries (earnings, FOMC, CPI/COPOM) inflate IV into the event, then crush it. Selecting an expiry that *straddles* an event means paying (as a buyer) or collecting (as a seller) that event premium — and eating the IV crush afterward. Align your DTE bucket with whether you *want* event exposure, and check the term structure (front-month IV elevated = backwardation = near-term stress priced in).

---

## Honest caveats

- Delta-probability and expected-move bands are **risk-neutral approximations**, distorted by skew — not real-world odds.
- tastytrade's 45/50%/21 defaults are **backtest-derived heuristics**, regime-dependent, with a structural edge (VRP) that has compressed over time.
- Mid-price fills, tight spreads, and clean assignment outcomes are **not guaranteed**.
- 0DTE and naked short premium carry **fat-tailed, potentially account-ending** losses. Defined-risk structures exist for a reason.
- Illustrative figures above (the $100/30%/30-day EM example, the 0.05→0.95 delta swing) are **examples**, not predictions.

**Sources:** [OIC Options Education](https://www.optionseducation.org/) · [OIC — Rule of 16](https://www.optionseducation.org/news/understanding-the-rule-of-16-in-plain-terms) · [OIC — Options Exercise FAQ](https://www.optionseducation.org/referencelibrary/faq/options-exercise) · [projectoption — Expected Move](https://projectoption.com/expected-move-calculator) · [Schwab — Delta & Probability](https://www.schwab.com/learn/story/options-delta-probability-and-other-risk-analytics) · [Option Alpha — Delta for Probabilities](https://optionalpha.com/lessons/using-delta-for-probabilities) · [FINRA — Exercise Cut-Off Notice](https://www.finra.org/rules-guidance/notices/information-notice-020321) · [Cboe — Index Options Tax Treatment](https://www.cboe.com/tradable_products/index-options-benefits-tax-treatment/) · [Cboe — SPX 0DTE 62.4% Record](https://www.cboe.com/insights/posts/spx-0-dte-options-jump-to-record-62-share-in-august/) · [Cboe — State of the Options Industry 2025](https://www.cboe.com/insights/posts/the-state-of-the-options-industry-2025) · [Trader Central — 21-DTE / 50% research](https://traderc.com/21-dte-50-percent-profit-exit-options/) · [B3 — exercício automático](https://www.b3.com.br/pt_br/noticias/exercicio-de-opcoes-passa-a-ser-automatico-a-partir-de-maio.htm) · [B3 Educação — Opções](https://edu.b3.com.br/w/opcoes) · [opcoes.net.br](https://opcoes.net.br/) · [OpLab](https://oplab.com.br/)

---

> **Disclaimer:** Educational/research material only — **not investment advice**, a recommendation, or a solicitation. Options involve substantial risk including total loss, and (for naked/short positions) losses exceeding premium received. Numbers are illustrative; verify current contract specs, margin, taxes, and costs with your broker/exchange. Past performance and backtests do not predict future results. **No options strategy is "free money."**

**Keywords:** options execution, strike selection, expiry selection, expected move, rule of 16, delta probability, volatility skew, theta gamma tradeoff, days to expiration (DTE), 45 DTE, 21 DTE, 50% profit target, limit order, multi-leg combo, liquidity screening, open interest, slippage, breakeven, pin risk, assignment, early exercise, OCC auto-exercise, exercise-by-exception, 0DTE, gamma risk, SPX, SPXW, SPY, Section 1256 60/40, B3; *opções, seleção de strike, vencimento, movimento esperado, regra do 16, delta, skew de volatilidade, theta, gama, decaimento temporal, ordem limitada, liquidez, deslizamento (slippage), ponto de equilíbrio, risco de pin, exercício/atribuição, exercício automático, R$0,01, liquidação D+2, sexta-feira do vencimento, opções semanais, 0DTE, Ibovespa, B3, OpLab*.
