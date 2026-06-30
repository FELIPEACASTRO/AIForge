# Volatility, Income & Neutral Options Strategies

> Non-directional options playbook: long/short volatility, premium-selling income, and neutral structures — with payoff, Greeks, breakevens, management rules, and honest tail-risk warnings. Global (2024–2026), with Brazil/B3 notes. **Not investment advice — research/education only.**

This page covers strategies whose primary bet is on **volatility** (vol) and **time decay** (theta), not on price direction (direção). The recurring economic edge for *selling* premium is the **variance risk premium (VRP)** — implied volatility (volatilidade implícita, IV) tends to print above subsequently realized volatility (volatilidade realizada), so option sellers are paid for bearing crash/gap risk. That premium is **compensation for tail risk, not free money**: short-vol books make small gains often and rare large losses (the "picking up pennies in front of a steamroller" profile). Size accordingly.

---

## 1. Core concepts you must internalize first

| Concept | Meaning (PT) | Why it matters |
|---|---|---|
| Variance Risk Premium (VRP) | prêmio de risco de variância | IV > realized vol on average → systematic edge for sellers, but it has **declined** in modern markets (Dew-Becker & Giglio 2025). |
| IV Rank / IV Percentile | rank/percentil de IV | Entry timing. tastytrade default: **sell premium when IVR > 50**, prefer trades when IVR is elevated; lean to *debit/long-vol* when IV is low. |
| Delta as ≈ probability | delta ≈ prob. ITM | A **16-delta** short strike ≈ 1 std dev OTM ≈ ~84% chance of expiring OTM → common strike-selection anchor. |
| Probability of Profit (POP) | probabilidade de lucro | Credit trades have high POP but **payoff is asymmetric** (small win, large max loss). High POP ≠ positive expectancy after costs. |
| Theta | decaimento temporal | Premium sellers are **long theta** (collect decay); buyers are short theta. |
| Vega | sensibilidade à IV | Sellers are typically **short vega** (hurt by IV spikes / "IV crush" works *for* them after events); calendars are **long vega**. |
| Gamma | gama | Risk accelerates near expiry/ATM. **Short gamma + short vega = the dangerous quadrant** (short straddles, 0DTE). |

**Management heuristics (tastylive "mechanics"), widely cited but not guarantees:**
- Enter defined-risk neutral spreads near **~45 DTE** (days to expiration / dias até o vencimento).
- Take profit at **~50% of max credit** for premium-selling trades.
- **Manage at 21 DTE** — close/roll regardless of P/L to dodge end-of-life gamma risk.
- Pick short strikes near **16-delta** (strangle) or **~25-delta** (iron condor wings) for balance of POP vs credit.

> These are *research-derived defaults* from tastylive backtests, not laws. Backtests are regime-dependent; the VRP edge has compressed since ~2010.

---

## 2. Long volatility (buying vol — defined risk, negative theta)

You **pay** premium, want a **big move** (either direction) and/or **rising IV**. You fight time decay every day.

| Strategy | Structure | Market view | Max profit | Max loss | Greeks | When to use |
|---|---|---|---|---|---|---|
| **Long Straddle** | Buy ATM call + ATM put, same strike & expiry | Big move, direction unknown; expect IV up | Unlimited (up) / large (down) | Debit paid (both premiums) | + vega, + gamma, **− theta** | Pre-event when IV is *low/underpriced*; binary catalysts |
| **Long Strangle** | Buy OTM call + OTM put (call strike > put strike) | Same as straddle, cheaper | Unlimited / large | Debit paid | + vega, + gamma, − theta | Cheaper than straddle; needs a *larger* move to pay |
| **Long Calendar** | Sell near-term + buy longer-term, **same strike** | Quiet now, **IV rises** later; pin near strike | Limited (peaks at strike) | Net debit | **+ vega**, + theta-ish near strike | Low IV environments; play a vol expansion / term-structure |

Breakevens (straddle): `strike ± total debit`. Strangle: `call strike + debit` and `put strike − debit`.

**Honest warning:** long vol bleeds theta daily and dies on **IV crush** — after earnings/events IV collapses and the long straddle can lose even if the stock moves "enough," because you overpaid for IV. Buy vol only when IV is genuinely cheap (low IVR) and the catalyst is real.

---

## 3. Short volatility (selling vol — INCOME, often UNDEFINED risk)

You **collect** premium, want the underlying to **stay in a range** and/or **IV to fall**. You are short gamma/vega — the painful quadrant.

| Strategy | Structure | Market view | Max profit | Max loss | Greeks | When to use |
|---|---|---|---|---|---|---|
| **Short Straddle** | Sell ATM call + ATM put, same strike | Tight range, falling IV | Credit received | **UNDEFINED** (uncapped up, large down) | − vega, **− gamma**, + theta | High IVR, expect mean reversion. **Margin-intensive, tail risk.** |
| **Short Strangle** | Sell OTM call + OTM put (e.g., 16-delta each) | Range-bound, falling IV | Credit received | **UNDEFINED** both sides | − vega, − gamma, + theta | Classic tastylive premium trade; **needs defined sizing & buying power buffer** |
| **Iron Condor** | Short strangle **+ long wings** (sell put spread + sell call spread) | Range-bound; want **defined risk** | Net credit | (Wing width − credit) × 100 | − vega, − gamma, + theta | Same view as short strangle but **capped loss**; the workhorse income trade |
| **Iron Butterfly** | Sell ATM call+put, buy OTM wings | Pin near a price; higher credit, narrower range | Net credit (larger than IC) | Wing width − credit | − vega, − gamma, + theta | Stronger pin conviction; tighter profit zone, bigger credit |
| **Broken-Wing Butterfly / Condor** | Skew one wing wider/closer to remove risk on one side | Neutral-to-directional lean | Credit (often) | Reduced/zero on one side | − vega, + theta | Take in a credit while eliminating risk on the side you don't fear |

**Iron condor math (the one to memorize):** with equal wing widths `W` and net credit `C` (per share): max profit = `C`; max loss = `W − C`; breakevens = `short put strike − C` and `short call strike + C`. Risk/reward is structurally lopsided (you risk much more than you can make) — POP must stay high to justify it.

> **Undefined-risk reality check (short straddle/strangle):** a single overnight gap (gap de abertura) or vol spike can erase months of credits. These are not "safe income." Always know your loss at a 2–3σ move, keep large buying-power reserves, and avoid concentrating in one underlying. Defined-risk variants (iron condor/butterfly) exist precisely to cap this.

### Income overlays on stock you own / want to own

| Strategy | Structure | Market view | Max profit | Max loss | When to use (PT) |
|---|---|---|---|---|---|
| **Covered Call** (lançamento coberto) | Long 100 shares + sell 1 OTM call | Neutral-to-mildly-bullish | (Strike − cost) + premium | Stock to zero (minus premium) | Generate yield on holdings; **caps upside** if called away |
| **Cash-Secured Put** (venda de put coberta por caixa) | Sell OTM put, hold cash = strike×100 | Neutral-to-bullish; willing to own | Premium | (Strike − premium) × 100 if → 0 | Get paid to set a buy limit; assignment = you buy the stock |
| **The Wheel** (a "roda") | CSP → if assigned, sell covered calls → if called away, repeat | Long-term bullish on a name you'd hold | Sum of premiums + cap gains | Full downside of owning the stock | Systematic income on quality names; **concentration risk** if all-in on one ticker |
| **Credit Put Spread** (trava de baixa em put / bull put) | Sell put + buy lower put | Neutral-to-bullish | Net credit | Width − credit | Defined-risk bullish income |
| **Credit Call Spread** (bear call) | Sell call + buy higher call | Neutral-to-bearish | Net credit | Width − credit | Defined-risk bearish income |
| **Jade Lizard** | Sell OTM put + sell OTM call spread | Neutral-to-slightly-bullish, high IV | Total credit | Downside (put) only — **no upside risk if credit ≥ call-spread width** | Coined by Dierking/Andrews on tastylive; removes upside tail by design |

Wheel/CSP/covered-call note: **assignment (exercício/atribuição) is a feature, not a bug** — but only trade them on names you genuinely want to own at the strike. Covered calls do not protect downside; they only soften it by the premium. The Wheel's worst case is the stock cratering while you keep getting assigned and selling calls below your cost basis.

---

## 4. Time/vol structures (calendars, diagonals, ratios)

| Strategy | Structure | Market view | Risk profile | Greeks | When to use |
|---|---|---|---|---|---|
| **Long Calendar** | Sell front-month, buy back-month, **same strike** | Quiet near-term, **rising IV / term-structure play** | Defined (debit) | **+ vega**, profits from front decay | Low IV now, expect expansion; pin near strike |
| **Diagonal Spread** | Like calendar but **different strikes** (vertical + calendar blend) | Mild directional + time/vol | Defined (debit or credit) | + vega (long), + theta lean | "Poor man's covered call" (long-dated call + short near call) |
| **Double Diagonal** | Diagonal on both put and call side | Range-bound + long vega | Defined | + vega, + theta | Income with a vol-expansion hedge baked in |
| **Ratio Spread** | Buy 1, sell 2 (e.g., 1×2 call ratio) | Directional with a vol view; often a credit | **Undefined on the naked side** | mixed; − gamma on excess shorts | Advanced; the extra short leg reintroduces undefined risk |
| **Back Spread** | Sell 1, buy 2 (inverse ratio) | Expect a *large* move | Defined; small loss if stagnant | + gamma, + vega | Convex long-vol with reduced cost |

**Calendar caveat:** it is **long vega** — a *drop* in IV after entry hurts even if price behaves. And because legs are different expiries, the payoff diagram is a snapshot, not fixed to expiration.

---

## 5. 0DTE income (SPX/SPY) — high reward, high gamma

Zero-days-to-expiration options reached **~59% of full-year 2025 SPX option volume** (with later months pushing above 60%, per CBOE). Traders sell same-day iron condors, credit spreads, and butterflies to harvest intraday theta.

**Why people do it:** all decay happens that day; defined-risk spreads (iron condor / credit spread) cap loss; lots of POP-friendly setups.

**Why it bites — gamma (gama):** at 0DTE, gamma is enormous near the money. A ~1% index move can swing a short strike's delta from ~0.05 to ~0.95 within hours, turning a "safe" condor into max loss almost instantly. There is **no time for the trade to recover or for IV mean-reversion to help.** Academic work (Dim, Eraker & Vilkov 2024) studies how dealer 0DTE gamma inventory amplifies intraday momentum/reversal and volatility propagation. CBOE's own research examines 0DTE's market-impact ("gamma squeezes").

> 0DTE is **active day-trading**, not passive income. Use strictly defined-risk structures, hard stops, small size, and accept that win-rate optics hide fat-tailed loss days.

---

## 6. Dispersion trading (index vs single-name vol) — advanced

**Long dispersion = short index volatility + long single-stock volatility.** It monetizes the **correlation risk premium**: index implied vol embeds correlation, so index options are often "expensive" vs the basket of single-name options. The trade profits when realized correlation stays *low* (names move idiosyncratically) and **loses sharply in crises when correlations spike to 1** (everything sells off together). CBOE now publishes the **DSPX** dispersion index. This is an institutional, capital- and execution-intensive strategy — listed here for completeness, not as a retail recommendation.

---

## 7. Consolidated risk register (read before trading)

| Risk | What happens | Mitigation |
|---|---|---|
| **Tail / gap risk** | Short straddle/strangle blows up on a gap or vol spike | Prefer defined-risk (condor/spread); size for 3σ; reserve buying power |
| **Short gamma near expiry** | Loss accelerates non-linearly (0DTE worst) | Manage at 21 DTE; avoid holding naked shorts to expiry |
| **IV crush** | Long options lose value as IV collapses post-event | Don't *buy* vol when IVR is high; sell events instead |
| **Assignment / early exercise** | Short ITM options (esp. before dividends) assigned early | Trade names you'd own (CSP/CC); watch ex-div dates; cash buffer |
| **Liquidity** | Wide bid/ask, bad fills, can't exit | Trade liquid underlyings; check open interest/volume; use limit orders |
| **Transaction costs** | Multi-leg credits eaten by commissions/slippage | Count costs in expectancy; high-POP small-credit trades are fragile to fees |
| **Pin risk** | Price sits exactly at short strike at expiry | Close before expiry; avoid 0DTE pins |
| **VRP decay** | The historical edge has shrunk | Don't assume backtests persist; treat short vol as risk-taking, not arbitrage |

---

## 8. Brazil / B3 specifics (🇧🇷)

- Liquidity concentrates in **opções sobre ações** (PETR4, VALE3, BBAS3, BOVA11) and **opções sobre o Ibovespa**; weeklies/0DTE depth is far thinner than US — **liquidity risk is real**; favor the most-traded series.
- **Lançamento coberto** (covered call) is the most popular income strategy with Brazilian retail; **venda de put coberta por caixa** (cash-secured put) and the *roda* (Wheel) are growing.
- Watch **exercício/atribuição** (American-style equity options can be exercised early, esp. around dividends/JCP) and **tributação** (IR on options/day-trade differs from spot — consult current B3/Receita rules).
- Tools: **[OpLab](https://oplab.com.br/)** (analysis/simulation, IV history, lançamento-coberto screens — "Taxa %", "Proteção", break-even), **[opcoes.net.br](https://opcoes.net.br/)** (chains, payoff, strategy builder), and **[B3 Educação — Opções](https://edu.b3.com.br/w/opcoes)** for official mechanics.

---

## 9. Where to learn & systematize (verified sources)

**Education / mechanics**
- CBOE / OIC — [all strategies](https://www.optionseducation.org/strategies/all-strategies-en), [short (iron) condor](https://www.optionseducation.org/strategies/all-strategies/short-condor) (note: OIC's "short condor" is a long-volatility, big-move trade — *not* the tastytrade short-premium iron condor described above), [volatility strategies](https://www.optionseducation.org/videolibrary/volatility-strategies)
- Fidelity options strategy guide — [long calendar (calls)](https://www.fidelity.com/learning-center/investment-products/options/options-strategy-guide/long-calendar-spread-calls), [long diagonal (calls)](https://www.fidelity.com/learning-center/investment-products/options/options-strategy-guide/long-diagonal-spread-calls), [double diagonal](https://www.fidelity.com/learning-center/investment-products/options/options-strategy-guide/double-diagonal-spread), [short iron condor](https://www.fidelity.com/learning-center/investment-products/options/options-strategy-guide/short-iron-condor-spread)
- Option Alpha — [Wheel strategy guide](https://optionalpha.com/blog/wheel-strategy)
- tastytrade — [platform glossary](https://support.tastytrade.com/support/s/solutions/articles/43000435395), [IVR on chart](https://support.tastytrade.com/support/s/solutions/articles/43000567353)
- Charles Schwab — [three things about the Wheel](https://www.schwab.com/learn/story/three-things-to-know-about-wheel-strategy)
- TradeStation — [Jade Lizard](https://www.tradestation.com/learn/options-education-center/the-jade-lizard-strategy-trading-sideways-and-rising-markets/)

**Research / papers (existence verified)**
- Dim, Eraker & Vilkov, *0DTEs: Trading, Gamma Risk and Volatility Propagation* (SSRN, 2023, R&R at *RFS*) — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4692190
- CBOE, *0DTE Index Options and Market Volatility: How Large is Their Impact? ("gamma squeezes")* — https://cdn.cboe.com/resources/education/research_publications/gammasqueezes.pdf
- Han & Zhou, *Variance Risk Premium and Cross-Section of Stock Returns* (SSRN, 2011) — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1785540
- Dew-Becker & Giglio, *The Decline of the Variance Risk Premium: Evidence from Traded and Synthetic Options* (SSRN/Chicago Fed WP 2025-17, 2025) — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5525882
- Heston & Todorov, *Exploring the Variance Risk Premium Across Assets* (a.k.a. *The Variance and Return Premiums across Assets*) (SSRN, 2023) — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4373509
- Quantpedia — [Volatility/Variance Risk Premium effect](https://quantpedia.com/strategies/volatility-risk-premium-effect), [Dispersion trading](https://quantpedia.com/strategies/dispersion-trading)
- CBOE — [DSPX S&P 500 Dispersion Index](https://www.cboe.com/us/indices/dispersion/) (launched Sept 2023), [benchmark indices](https://www.cboe.com/us/indices/benchmark_indices/)

**Book**
- Euan Sinclair, *Positional Option Trading: An Advanced Guide* (Wiley, 2020) — vol premium, term-structure, earnings edges, risk & sizing — https://www.wiley.com/en-us/Positional+Option+Trading%3A+An+Advanced+Guide-p-9781119583530

**Tooling / code (verified repos)**
- [optopsy](https://github.com/goldspanlabs/optopsy) — options backtesting (condors, butterflies, calendars, diagonals, covered positions)
- [OptionSuite](https://github.com/sirnfs/OptionSuite) — options/stock backtester & live-trade framework
- [Option-strategies-backtesting-in-Python](https://github.com/OptionsnPython/Option-strategies-backtesting-in-Python) — Greeks + strategy backtests
- [backtesting.py](https://github.com/kernc/backtesting.py) — general Python backtesting engine

---

> **Disclaimer / Aviso:** Educational and research material only — **not investment advice** (não é recomendação de investimento). Options involve substantial risk, including undefined and total loss of capital; short-volatility strategies carry tail/gap risk that can exceed deposited funds. Past backtests and the variance risk premium do not guarantee future results. Verify current B3/Receita rules and consult a licensed professional before trading.

**Keywords:** options strategies, volatility trading, premium selling / venda de prêmio, theta harvesting / colheita de theta, variance risk premium / prêmio de risco de variância, iron condor (condor de ferro), iron butterfly (borboleta de ferro), short strangle / strangle vendido, long straddle (compra de straddle), calendar spread / trava calendário, diagonal spread, ratio spread, covered call / lançamento coberto, cash-secured put / venda de put coberta por caixa, the Wheel / a roda, credit spread / trava de crédito, jade lizard, 0DTE, gamma risk / risco de gama, IV rank / rank de volatilidade implícita, 16-delta, probability of profit / probabilidade de lucro, dispersion trading, B3, Ibovespa, OpLab, opções
