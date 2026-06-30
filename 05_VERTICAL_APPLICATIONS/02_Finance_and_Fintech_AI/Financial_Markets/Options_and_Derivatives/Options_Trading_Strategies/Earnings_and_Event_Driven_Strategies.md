# Earnings & Event-Driven Options Strategies

> Trading options around **discrete, scheduled catalysts** — earnings, ex-dividend, M&A, FDA/biotech binaries, index rebalances, product launches. Focus on the **IV ramp-up → IV crush** cycle, the **expected move** vs the realized move, and binary-event sizing. The honest verdict up front: earnings moves are near-random in direction, IV crush cuts both ways, and the edge is thin. **Research/education only — not investment advice.**

This page complements the sibling playbook pages (Strategy Selection; Directional & Spreads; Volatility/Income/Neutral; Position Management; Systematic & ML; Hedging/Tail/Macro). It does **not** re-explain payoff diagrams of straddles or condors — see *Volatility, Income & Neutral*. Here the angle is **the event itself**: how implied volatility (volatilidade implícita, IV) behaves around a known date, how to price the implied move, and which structures monetize each event's specific mechanics.

---

## 1. The core event microstructure: IV ramp-up and IV crush

Before a scheduled catalyst, **front-expiry IV rises** ("vol ramp", *empilhamento de IV*) because the option must price a known discontinuity. After the number prints, the uncertainty resolves in an instant and **IV collapses** — the **IV crush** (*esmagamento de IV* / *vol crush*). This is the single most important phenomenon on this page.

| Phase | What happens to IV | Who benefits | Who is hurt |
|---|---|---|---|
| Days/weeks before event | Front-expiry ATM IV climbs; term structure inverts (front IV > back IV) | Already-long-vega holders (mark-to-market) | New long-premium buyers pay up |
| The print (overnight gap) | Realized one-day move resolves; IV begins to deflate | Whoever guessed direction/magnitude correctly | The other side |
| The open after | IV crushes hard (often 20–50% relative drop in front IV) | **Short premium / short vega** | **Long premium / long vega** |

**Magnitudes (illustrative, source-cited):** ORATS-style studies and Market Chameleon's per-ticker pages routinely show single-name front IV30 falling **~20–25% on the day after earnings** ([Market Chameleon earnings charts](https://marketchameleon.com/Overview/MU/Earnings/Earnings-Charts/)); biotech PDUFA names can carry **150–300% annualized IV** pre-decision that craters post-decision ([Benzinga FDA plays](https://www.benzinga.com/pro/blog/how-to-trade-biotech-stocks-strategies-and-tools-for-fda-plays)). Treat any specific percentage as ticker- and regime-dependent.

> **The trap that defines this page:** you can be *right on direction* and still lose money because IV crush deflated your long option faster than the move inflated it. Conversely, you can sell premium, watch IV crush in your favor, and still lose if the gap blows through your strikes. Both errors are common.

---

## 2. The expected move (implied move) — how to compute it

The **expected move** (*movimento esperado / implícito*) is the options-market's ~1 standard-deviation price range through the event. It is the benchmark a long-premium buyer must *beat* and a short-premium seller hopes to *contain*.

| Method | Formula | Notes |
|---|---|---|
| **IV (closed-form 1σ)** | `EM ≈ Spot × IV × √(DTE/365)` | Generic; uses annualized ATM IV and calendar days. ~68% containment. |
| **ATM straddle shortcut** | `EM ≈ 0.80–0.85 × (ATM straddle price)` of the expiry **immediately after** the event | The ~0.8 factor (some desks divide the straddle by 1.25, i.e. ×0.80) trims the straddle's slight overshoot of 1σ. Fast desk heuristic ([MenthorQ](https://menthorq.com/guide/from-straddle-price-to-expected-move/)). |
| **tastytrade weighted blend** | `EM = 0.60×ATM straddle + 0.30×1st OTM strangle + 0.10×2nd OTM strangle` | The platform's published method ([tastytrade EM article](https://support.tastytrade.com/support/s/solutions/articles/43000435415)). |

Use the **expiry that expires just after the announcement** to isolate the event premium; the post-event weekly straddle is dominated by the earnings jump, not ordinary time premium. The earnings expected move is typically **2–4× the ordinary weekly move** for the same name ([projectfinance](https://www.projectfinance.com/expected-move/), [Options Hawk](https://optionshawk.com/calculating-expected-moves-using-options/)).

**Worked example (illustrative):** Stock at \$230, post-earnings weekly straddle = \$14.00 → EM ≈ \$14 × 0.85 ≈ **\$11.90** (~5.2%). A long straddle here needs the stock outside **\$218.10 / \$241.90** *at exit* just to break even **after** IV crush — a higher bar than \$11.90 of raw movement, because the residual options will have lost their event vol.

---

## 3. Implied vs historical (realized) move — where any edge lives

The only repeatable edge in earnings options is a **persistent gap between implied and realized** moves on a given ticker. Tools that backtest this per symbol:

| Tool | What it gives you | URL |
|---|---|---|
| **Market Chameleon** | Implied vs actual move over last ~12 quarters; IV crush %; pre-set earnings strategy win rates | [marketchameleon.com](https://marketchameleon.com/Overview/TRUE/Earnings/Earnings-Charts/) |
| **ORATS** | Earnings Move Report; custom backtester over 5,000+ symbols / 25 strategies; implied vs avg actual move | [orats.com/backtester](https://orats.com/backtester) |
| **tastytrade** | Expected-move overlay on the platform; market-study research | [tastylive.com](https://www.tastylive.com/) |
| **CBOE / OIC** | Authoritative mechanics, assignment & dividend education (not signals) | [optionseducation.org](https://www.optionseducation.org/) |

**Reading the signal:** If a name has *consistently moved less than implied* (e.g., META in an ORATS dashboard showing implied 6.8% vs 12-quarter average actual 13.2% — here implied is *below* realized, favoring buyers; the reverse favors sellers — [ORATS via Nasdaq](https://www.nasdaq.com/articles/will-meta-surprise-again-orats-data-shows-traders-are-bracing-volatility)), that is a *data point*, not a guarantee. Earnings regimes shift; a name that under-delivered eight times can gap 3× implied on the ninth.

---

## 4. The two earnings camps: long premium vs short premium

| | **Long premium (buy vol)** | **Short premium (sell vol)** |
|---|---|---|
| Structures | Long straddle / strangle; (sometimes) calendars in reverse | Short straddle / strangle, **iron condor**, **iron butterfly** |
| Bet | Realized move **>** expected move **and/or** you overcome IV crush | Realized move **<** expected move; **harvest the IV crush** |
| Greeks into event | +vega, +gamma, **−theta**; hurt by crush | **−vega**, −gamma, +theta; helped by crush |
| Wins when | Big surprise gap that exceeds the implied move | Quiet print; stock stays inside the implied range |
| Loses when | "In-line" result → crush evaporates premium even on a decent move | Gap blows through strikes (undefined risk on naked) |
| Profile | Many small losses, rare large win (lottery-ish) | Many small wins, rare large loss (negative skew) |

**Empirical reality check (cite, don't trust blindly):**
- ORATS' earnings backtest (**5,217 announcements, 20,868 trades**; in-sample Jan-2020→Jul-2021, out-of-sample Jul-2021→Oct-2021): in-sample **Sell Straddle +1.18%** over 214 trades but with a worst trade of **−26.7%** vs best **+8.3%** — textbook negative skew; **Buy Straddle +0.40%** (worst −3.8%, best +13.7%); the **long calendar** was the best out-of-sample performer (+0.91%) ([ORATS backtest](https://orats.com/blog/earnings-options-strategies-backtest)).
- An independent study across 4,200 events reports an **average IV crush of ~38%** and a **next-open short-straddle win rate of 54.7%** with avg win +19.4% / avg loss −22.1% — i.e., barely-above-coin-flip hit rate paired with *unfavorable* payoff asymmetry ([iPresage research](https://www.ipresage.com/research/earnings-iv-crush)).

The takeaway both camps must internalize: **after costs and skew, expected value hovers near zero.** Any positive expectancy comes from disciplined ticker selection (implied-vs-realized edge), sizing, and management — not from the structure itself.

---

## 5. Calendars & diagonals — exploiting the term-structure inversion

Because front-expiry IV spikes *more* than back-expiry IV into the event, the term structure inverts. **Calendars/diagonals sell the rich front and buy the cheaper back**, monetizing the differential crush.

| Structure | Build | Why it fits earnings | Main risk |
|---|---|---|---|
| **Long calendar** | Sell front-week ATM, buy next-cycle same strike | Front IV crushes harder than back → vega differential pays | A move that overwhelms the strike (gap away from the pin) |
| **Double calendar** | Two calendars at a call strike and a put strike straddling spot | Wider profit zone than single calendar | Same gap risk; net **debit** at stake |
| **Diagonal** | Calendar with different strikes (directional lean) | Adds a directional tilt to the vol-differential play | Both vol and direction must cooperate |

Mechanics & timing: enter **1–5 days before** the print when the IV differential is most pronounced (entering 2–3 weeks early bleeds theta with little differential yet); max profit when the stock **pins near the short strike** at front expiry ([OptionsTradingIQ double-calendar backtest](https://optionstradingiq.com/double-calendar-earnings-trade/), [strike.money](https://www.strike.money/options/double-calendar-spread)). These are **defined-risk debit** trades — you cannot lose more than the debit, which makes them a popular "I want event vol exposure without uncapped tail" choice.

---

## 6. Ex-dividend & early assignment of ITM calls (a non-earnings event)

For **American-style** options (all single-name US equity options; B3 equity options are predominantly American, especially calls), the dominant rational early-exercise case is a **deep-ITM call before an ex-dividend date**. If you are **short** that call, you can be assigned, go short the stock over the ex-date, and **owe the dividend**.

**Decision rule (memorize):** a long call holder rationally exercises early to capture a dividend when the call's **remaining extrinsic value (time + IV premium) < the dividend**. So as a short-call writer:

> If your short call is ITM and its **extrinsic value is less than the upcoming dividend**, **expect assignment** the day before ex-date.

| Risk factor | Direction of effect | Source |
|---|---|---|
| Call moneyness ↑ (deeper ITM) | Assignment risk ↑ | [Schwab](https://www.schwab.com/learn/story/ex-dividend-dates-understanding-dividend-risk) |
| Time to expiry ↓ | Extrinsic ↓ → assignment risk ↑ | [OIC / optionseducation.org](https://www.optionseducation.org/referencelibrary/faq/options-assignment) |
| Dividend size ↑ | Assignment risk ↑ (more to capture) | [Fidelity](https://www.fidelity.com/learning-center/investment-products/options/dividends-options-assignment-risk) |
| Extrinsic value > dividend | Assignment **unlikely** | [Option Alpha](https://optionalpha.com/learn/dividend-assignment-risk) |

**Mitigation:** before ex-date, **buy back** the threatened short call or **roll** it up/out to restore extrinsic value above the dividend. Assignment can arrive **days or weeks early**, and being assigned short stock triggers extra margin plus the dividend liability ([OIC exercising options](https://www.optionseducation.org/optionsoverview/exercising-options)). **Put** early-exercise mirrors this (deep-ITM puts just *after* ex-date), but the dividend-driven call case is the everyday hazard.

---

## 7. Other discrete events

| Event | Typical strategy | Rationale | Risk |
|---|---|---|---|
| **M&A / deal (cash)** | Often **stock** arb; options to cap downside — buy OTM put on target / OTM call on acquirer | ~90–95% of announced deals close → target trades at a discount to offer; the spread is the carry | Deal break → target gaps **−20% to −40%**; binary, fat left tail ([M&I](https://mergersandinquisitions.com/merger-arbitrage/), [AnalystPrep](https://analystprep.com/study-notes/cfa-level-2/event-driven-strategies-merger-arbitrage/)) |
| **M&A (stock-for-stock)** | Long target / short acquirer at the exchange ratio; options to bound | Captures spread without naked short-sale constraints | Ratio risk, collar adjustments, regulatory block |
| **FDA / biotech (PDUFA, trial readout)** | Defined-risk: **long calls/puts** or debit spreads/strangles to cap loss at premium | True binary; stock can move **40–200%+** overnight in either direction | Pre-event IV **150–300%** → brutal crush; pre-PDUFA "run-up" can fade; correct direction can still lose to crush ([Benzinga](https://www.benzinga.com/pro/blog/how-to-trade-biotech-stocks-strategies-and-tools-for-fda-plays), [Dan Sfera PDUFA](https://dansfera.com/pdufa-explained)) |
| **Index rebalance / reconstitution** | Trade the add/delete name or the closing auction; options for cheap exposure | "Index effect": adds get index-fund buying, deletes get selling; volume spikes hugely into the close, then partially reverses | Effect is short-lived and increasingly arbitraged away; reversal by next open ([CME OpenMarkets](https://www.cmegroup.com/openmarkets/equity-index/2025/Navigating-the-S-P-500-Rebalance-A-Quarterly-Market-Ritual.html)) |
| **Product launch / guidance / investor day** | Treat like a soft earnings event: price the implied move, lean long or short vol | Smaller, fuzzier vol ramp than earnings | "Sell-the-news" reversals; low signal-to-noise |
| **Macro prints (CPI, FOMC, NFP)** | Index/ETF straddles or 0DTE; or short premium to harvest the event vol ramp | Scheduled, market-wide vol ramp & crush | Whipsaw; 0DTE gamma risk — see *Position Management* page |

> Merger arb and biotech binaries are **fundamentally not vol trades** — they are *probability-of-outcome* trades wearing an options costume. Options here are mainly for **defining/capping risk** on a binary, not for harvesting theta.

---

## 8. Position sizing for binary events

Binary catalysts can gap *through* every strike overnight; gamma scalping and "manage at 21 DTE" mechanics do **not** save you across an event gap. Size as if the worst plausible gap happens **tonight**.

| Principle | Practical rule of thumb |
|---|---|
| **Define max loss before entry** | Prefer defined-risk structures (spreads, debit calendars, long premium) for true binaries; the loss = the debit/width. |
| **Size by max loss, not margin** | Risk a **small fixed fraction** of capital per binary event (many desks use ~1–2%); never size off "probability of profit" alone — POP is high precisely when payoff is worst. |
| **Respect negative skew** | Short-premium earnings trades win often but lose big; one −300% straddle erases many +50% winners. Cap per-name exposure. |
| **Avoid concentration in correlated events** | Multiple biotechs into the same FDA cycle, or a sector's earnings on one night, are *one* bet, not many. |
| **Liquidity & costs first** | Wide event-week spreads and skew make slippage a real, recurring tax — the thin edge often lives entirely inside the bid/ask. |

---

## 9. B3 (Brazil) specifics

- **American-style equity options** (*opções de ações americanas*): PETR4, VALE3, etc. can be **exercised any time before expiry** → the ex-dividend early-assignment logic in §6 **applies directly**. Watch **JCP/dividendos** (*juros sobre capital próprio*) ex-dates on short ITM calls.
- **Expiration**: monthly equity-option expiry is the **3rd Friday** (rule effective since May 2021; previously the 3rd Monday); **weekly options** (*opções semanais*) expire every Friday **except** the 3rd ([B3 weekly options](https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/renda-variavel/opcoes-semanais-sobre-acoes.htm), [B3 expiry calendar](https://www.b3.com.br/pt_br/solucoes/plataformas/puma-trading-system/para-participantes-e-traders/calendario-de-negociacao/vencimentos/)). Automatic exercise applies when ITM by ≥ R\$0.01 at the close.
- **Earnings (*balanços / resultados*)** cluster on a published B3 calendar; liquid single-name option chains are concentrated in the largest names (PETR4, VALE3, BOVA11, ITUB4, B3SA3), so per-ticker implied-vs-realized history is thinner than in US names — calibrate carefully.
- **Liquidity caveat:** outside the top tickers and the front-month, B3 option spreads widen fast; event-week slippage can dominate the trade's expectancy.

---

## 10. Honest risk summary

- **Direction is near-random.** The earnings *gap sign* is close to a coin flip; do not confuse a good thesis with edge.
- **IV crush cuts both ways.** It is the seller's friend and the buyer's enemy — but a large enough gap reverses both.
- **Edge is thin and fragile.** Published backtests show near-zero average EV after skew; the only durable angle is *ticker-level implied-vs-realized mispricing*, disciplined sizing, and cost control.
- **Costs & assignment are real.** Wide event spreads, dividend early-assignment, and pin risk at expiry quietly erode results.
- **PEAD exists but is not a free lunch.** Post-earnings announcement drift (*deriva pós-anúncio*) is documented (Govindaraj, Liu & Livnat, [SSRN 2146181](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2146181)) and option-implied vols carry information about it, but it is small, decays, and is heavily arbitraged.

> **Not investment advice.** Educational/research synthesis only. Options can lose 100% of premium (long) or expose undefined loss (naked short). Validate every mechanic against your broker and a primary source before risking capital.

---

**Sources:** [CBOE/OIC optionseducation.org](https://www.optionseducation.org/) · [OIC options assignment FAQ](https://www.optionseducation.org/referencelibrary/faq/options-assignment) · [tastytrade expected move](https://support.tastytrade.com/support/s/solutions/articles/43000435415) · [tastylive](https://www.tastylive.com/) · [Option Alpha — dividend assignment risk](https://optionalpha.com/learn/dividend-assignment-risk) · [The Options Playbook — early exercise](https://www.optionsplaybook.com/managing-positions/early-options-exercise) · [Market Chameleon earnings/IV-crush](https://marketchameleon.com/Overview/TRUE/Earnings/Earnings-Charts/) · [ORATS earnings backtest](https://orats.com/blog/earnings-options-strategies-backtest) · [ORATS backtester](https://orats.com/backtester) · [Fidelity — dividends & assignment](https://www.fidelity.com/learning-center/investment-products/options/dividends-options-assignment-risk) · [Schwab — ex-dividend risk](https://www.schwab.com/learn/story/ex-dividend-dates-understanding-dividend-risk) · [MenthorQ — straddle→expected move](https://menthorq.com/guide/from-straddle-price-to-expected-move/) · [projectfinance — expected move](https://www.projectfinance.com/expected-move/) · [OptionsTradingIQ — double calendar earnings](https://optionstradingiq.com/double-calendar-earnings-trade/) · [Benzinga — biotech/FDA plays](https://www.benzinga.com/pro/blog/how-to-trade-biotech-stocks-strategies-and-tools-for-fda-plays) · [M&I — merger arbitrage](https://mergersandinquisitions.com/merger-arbitrage/) · [AnalystPrep — event-driven/merger arb](https://analystprep.com/study-notes/cfa-level-2/event-driven-strategies-merger-arbitrage/) · [CME OpenMarkets — S&P rebalance](https://www.cmegroup.com/openmarkets/equity-index/2025/Navigating-the-S-P-500-Rebalance-A-Quarterly-Market-Ritual.html) · [Govindaraj/Liu/Livnat — PEAD & option traders, SSRN 2146181](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2146181) · [iPresage — IV crush study](https://www.ipresage.com/research/earnings-iv-crush) · [B3 weekly options](https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/renda-variavel/opcoes-semanais-sobre-acoes.htm) · [B3 expiry calendar](https://www.b3.com.br/pt_br/solucoes/plataformas/puma-trading-system/para-participantes-e-traders/calendario-de-negociacao/vencimentos/)

**Keywords:** earnings options, event-driven options, IV crush (esmagamento de IV), implied volatility ramp (empilhamento de IV), expected move / implied move (movimento esperado), ATM straddle (straddle no dinheiro), iron condor / iron butterfly (condor de ferro / borboleta de ferro), calendar & diagonal spreads (calendário / diagonal), ex-dividend early assignment (exercício antecipado / dividendos), short premium (venda de prêmio), binary event (evento binário), merger arbitrage (arbitragem de fusão), FDA/PDUFA biotech catalyst (catalisador biotecnologia), index rebalance (rebalanceamento de índice), PEAD (deriva pós-anúncio de resultados), position sizing (dimensionamento de posição), B3 opções, vencimento, balanços.
