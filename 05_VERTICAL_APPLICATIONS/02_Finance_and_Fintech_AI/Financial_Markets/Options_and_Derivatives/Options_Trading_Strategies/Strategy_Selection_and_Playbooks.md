# Options Strategy Selection & Playbooks

> A practical decision framework for choosing options strategies — mapping your market view × volatility view × time horizon to concrete buy-premium and sell-premium plays, with honest treatment of probability of profit, payoff asymmetry, assignment, and IV crush. Education/research only — **not investment advice**.

---

## 1. The Core Decision: Buy Premium or Sell Premium?

Every options decision reduces to two orthogonal questions, then a third:

1. **Direction** — Where do you think the underlying goes? (bullish / bearish / neutral)
2. **Volatility** — Is implied volatility (IV, *volatilidade implícita*) **cheap or rich**, and is it likely to **rise or fall**?
3. **Time** — How fast must the move happen vs. how much time decay (theta, *decaimento temporal*) you can fund or harvest.

The single most useful filter is **IV Rank (IVR)** / **IV Percentile** — where current IV sits relative to its own past year. tastylive's house rule of thumb: **sell premium when IVR > ~35–50, prefer buying/defined-risk when IVR is low**, because IV mean-reverts and a high starting point is a tailwind for sellers ([tastytrade volatility metrics](https://support.tastytrade.com/support/s/solutions/articles/43000539059)).

| | **Buy premium (long options)** | **Sell premium (short options)** |
|---|---|---|
| Net Greeks | Long vega, long gamma, **short theta** (pays decay) | Short vega, short gamma, **long theta** (collects decay) |
| Wants IV to | **Rise** (and start low/cheap) | **Fall** (and start high/rich) |
| Wants price to | Move **big and fast** | **Stay in a range / decay** |
| Probability of profit (POP) | Usually **low** (need to beat premium paid) | Usually **high** (range is wide) |
| Payoff shape | Small, defined cost → large/convex upside | **Large/undefined risk → small capped credit** |
| Best when | IVR low; pre-move; convexity wanted | IVR high; post-spike; range-bound; harvesting VRP |

> **The fundamental tradeoff (be honest):** high-POP strategies (selling premium) win *often* but each loss can dwarf many wins — "**high POP = small wins, large risk**." Low-POP strategies (buying premium) lose often but pay convexly when right. Neither is free money; the **expected value** after costs and slippage is what matters, and the structural edge sellers harvest is the **volatility risk premium (VRP)** — IV tends to print above subsequent realized vol, but it is *not* guaranteed and reverses violently in crashes.

> **Benchmarked reality check:** Cboe publishes long-history S&P 500 strategy-benchmark indices that track exactly these systematic plays — **BXM** (BuyWrite / covered call), **PUT** (PutWrite / cash-secured put), **CLL** (95-110 Collar), **CNDR** (Iron Condor), and **BFLY** (Iron Butterfly). They are useful for studying realized return/drawdown profiles before risking capital ([Cboe Strategy Benchmark Indices](https://www.cboe.com/us/indices/benchmark_indices/)).

---

## 2. Expected Value, POP, and Breakevens — the math you must respect

- **POP (probabilidade de lucro):** approx. probability the position is at/above breakeven at expiration. For a short OTM option, POP ≈ 1 − (delta of the short strike) is a rough proxy; brokers compute it from the option-implied distribution.
- **Expected value (EV):** `EV = POP × avg_win − (1−POP) × avg_loss − costs`. A 90%-POP credit spread that risks $9 to make $1 needs a >90% real win rate just to break even after commissions/slippage. **POP alone is meaningless without payoff and costs.**
- **Breakeven (ponto de equilíbrio):**
  - Long call: `strike + debit`. Long put: `strike − debit`.
  - Short put / cash-secured put: `strike − credit`. Covered call: `cost basis − credit`.
  - Credit spread: `short strike ± credit`. Long straddle: `strike ± total debit` (needs move > the implied move to profit).
- **Risk/reward vs. POP are inversely linked.** You cannot get high POP *and* favorable payoff *and* short duration simultaneously; pick two and pay for the third.

Practitioner sources like Option Alpha bias toward selling out-of-the-money strikes far enough from the money to keep probability of profit high, explicitly trading large-but-rare losses for frequent small wins ([Option Alpha — Bull Put Spread](https://optionalpha.com/strategies/bull-put-credit-spread); [SPY put-spread backtest](https://optionalpha.com/blog/spy-put-credit-spread-backtest)).

---

## 3. Master Selection Matrix (Direction × Volatility)

Recommended structures by view. Defined-risk variants in **bold** for capital/assignment safety.

| Market view ↓ \ Vol view → | **IV LOW / cheap (buy premium)** | **IV HIGH / rich (sell premium)** |
|---|---|---|
| **Bullish** | Long call; **bull call (debit) spread**; long-dated LEAPS call; risk reversal | **Bull put (credit) spread**; cash-secured put; covered call; ratio put spread |
| **Bearish** | Long put; **bear put (debit) spread**; protective put on longs | **Bear call (credit) spread**; covered put; call ratio spread |
| **Neutral / range** | Long **calendar/diagonal** (buy back month vs front), long butterfly | **Iron condor**; short strangle/straddle; **iron butterfly**; jade lizard |
| **Big move, unsure direction** | Long straddle / long strangle; **debit "double" spreads**; reverse calendar | (avoid selling) — undefined-risk if you sell into low IV |
| **Quiet drift / income** | Poor fit for buying | Covered call (*lançamento coberto*); cash-secured put; the "wheel" |

> Add the **time axis:** short DTE (0–7d) maximizes theta but gamma risk explodes near expiry; ~30–60 DTE is tastylive's common sweet spot for selling premium (enough theta, manageable gamma); buyers often go **longer** DTE to reduce theta bleed while waiting for the move.

---

## 4. Strategy Playbooks (structure · view · max P/L · Greeks · when)

### 4a. Long / debit (buying premium — IV low, expect fast move)

| Strategy | Structure | Market view | Max profit | Max loss | Breakeven(s) | Greeks profile | When to use |
|---|---|---|---|---|---|---|---|
| Long call | Buy 1 call | Strongly bullish | Unlimited | Debit paid | Strike + debit | +δ +Γ +Vega −Θ | IVR low, expected catalyst, want convex upside |
| Long put | Buy 1 put | Strongly bearish | Strike − debit (≫0) | Debit paid | Strike − debit | −δ +Γ +Vega −Θ | Cheap downside / crash hedge when IV is low |
| Bull call (debit) spread | Buy lower call, sell higher call | Moderately bullish | Width − debit | Debit paid | Long strike + debit | +δ, vega-light | Bullish but want lower cost / cap IV exposure |
| Bear put (debit) spread | Buy higher put, sell lower put | Moderately bearish | Width − debit | Debit paid | High strike − debit | −δ, vega-light | Bearish, cheaper than long put |
| Long straddle | Buy ATM call + put | Big move, no direction | Large/unlimited | Total debit | Strike ± debit | δ≈0 +Γ ++Vega −−Θ | Pre-event only if IV **not** already inflated |
| Long strangle | Buy OTM call + OTM put | Big move, cheaper | Large/unlimited | Total debit | Strikes ± debit | δ≈0 +Γ +Vega −Θ | Expect large move, lower cost than straddle |
| Calendar spread | Sell front, buy same-strike back month | Neutral now, vol up later | Limited (peaks at strike) | Net debit | Around strike | +Vega, +Θ-ish, short gamma front | Low front-month IV / contango term structure |

### 4b. Short / credit (selling premium — IV high, expect range/decay)

| Strategy | Structure | Market view | Max profit | Max loss | Breakeven(s) | Greeks profile | When to use |
|---|---|---|---|---|---|---|---|
| Cash-secured put | Sell 1 put (cash to buy 100sh) | Bullish/neutral | Credit | Strike − credit (large) | Strike − credit | +δ −Vega +Θ | High IVR, willing to **own** shares cheaper |
| Covered call | Long 100sh + sell 1 call | Neutral/mild bull | Credit + (strike−basis) | Basis − credit | Basis − credit | −δ vs stock, −Vega +Θ | Income on holdings; cap upside |
| Bull put (credit) spread | Sell higher put, buy lower put | Bullish/neutral | Credit | Width − credit | Short strike − credit | +δ −Vega +Θ | High IVR, defined risk, want high POP |
| Bear call (credit) spread | Sell lower call, buy higher call | Bearish/neutral | Credit | Width − credit | Short strike + credit | −δ −Vega +Θ | High IVR, resistance overhead |
| Iron condor | Bull put + bear call (both OTM) | Range-bound | Net credit | Width − credit | Short strikes ± credit | δ≈0 −Vega +Θ | **High IVR**, expect range; defined risk |
| Iron butterfly | ATM short straddle + OTM wings | Pin near strike | Net credit | Width − credit | Strike ± credit | δ≈0 −Vega +Θ | Very high IVR, strong pin view |
| Short strangle | Sell OTM call + OTM put (naked) | Range-bound | Net credit | **Undefined** both sides | Strikes ± credit | δ≈0 −−Vega +Θ −Γ | High IVR, high POP — **undefined risk** ⚠ |
| Short straddle | Sell ATM call + put (naked) | Strong pin | Net credit | **Undefined** | Strike ± credit | δ≈0 −−Vega +Θ −Γ | Max premium, max risk ⚠ |
| Jade lizard | Short put + short call spread (no upside risk) | Neutral/bull | Net credit | Put side large | Put strike − credit | −Vega +Θ | High IVR, slight bullish, no upper risk |

> ⚠ **Undefined-risk warning:** short strangles/straddles and naked calls can lose far more than the credit collected (a naked call's loss is theoretically **unlimited**). They require high margin, active management, and survive on tail discipline. Define risk with spreads/condors unless you fully understand and can fund the worst case.

---

## 5. The Role of IV Rank, Percentile & Term Structure

- **IV Rank** = `(IV_now − IV_low_1y) / (IV_high_1y − IV_low_1y)`. Sensitive to single spikes. **IV Percentile** = % of days in the past year IV was below today's — more robust to outliers. tastytrade publishes both (IVR, IV%, IVx) ([volatility metrics](https://support.tastytrade.com/support/s/solutions/articles/43000539059)).
- **Decision use:** high IVR/IV% → favor **selling**; low → favor **buying** or defined-risk debit structures. IVR can read >100 or <0 when current IV breaks its prior-year range ([tastytrade FAQ](https://support.tastytrade.com/support/s/solutions/articles/43000559231)).
- **Term structure (estrutura a termo da volatilidade):**
  - **Contango** (back months IV > front) = calm; favors **long calendars** (buy cheap back-month vega).
  - **Backwardation** (front IV > back) = near-term stress/event; favors selling the inflated front (reverse calendar / front-month credit) and is a flashing sign of an upcoming binary ([FlashAlpha — term structure](https://flashalpha.com/articles/volatility-term-structure-contango-backwardation-events)).

---

## 6. Event Trading (earnings) vs. Non-Event

**Before a known binary (earnings, FOMC, FDA):** IV inflates into the event, then **IV crush** (*esmagamento de volatilidade*) collapses it overnight. A long straddle can be **directionally right and still lose** because the vega loss exceeds the delta gain — front-month equity IV commonly drops 30–60% post-print ([SpotGamma — IV Crush](https://support.spotgamma.com/hc/en-us/articles/15249330755859-IV-Crush-Explained-What-It-Is-When-It-Happens-and-How-to-Trade-It)).

| Situation | Buyers | Sellers |
|---|---|---|
| Pre-earnings (IV inflated) | **Bad** — overpay for vega, crushed after | **Good** — collect rich premium, harvest crush (defined-risk: iron condor/short strangle managed) |
| Non-event, low IVR | **Good** — cheap convexity for a real catalyst | **Bad** — thin credit, no cushion |
| Post-spike (IV collapsing) | Bad | **Good** — sell into elevated then-decaying IV |

> Selling earnings premium has a **negative-skew payoff**: many small wins, occasional large gap losses when the move exceeds the implied move. Size accordingly.

---

## 7. Capital, Assignment & Friction (the realities)

- **Assignment (exercício/atribuição):** short options can be assigned; **short ITM American options can be exercised early**, especially short calls before ex-dividend and deep-ITM puts. Plan for share delivery and margin calls.
- **Liquidity:** trade only **tight bid/ask, high open-interest** chains. Wide spreads silently destroy EV; mid-price fills are not guaranteed.
- **Transaction costs:** commissions + slippage compound across multi-leg trades and frequent rolling — fatal to thin-credit, high-POP strategies.
- **Margin & buying power:** undefined-risk shorts demand large buying-power reduction; defined-risk spreads cap it at width − credit.
- **Management:** tastylive convention is taking credit trades off near **~50% of max profit** and managing/rolling tested sides rather than holding to expiry — reduces gamma risk and improves realized win rate.

---

## 8. 🇧🇷 Brazil / B3 Specifics

- **Products:** *opções sobre ações* (equity options, e.g. PETR4, VALE3, BBAS3) and **opções sobre o Ibovespa** (index options) on **B3** — see [B3 — Opções sobre Ibovespa](https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/renda-variavel/opcoes-sobre-ibovespa.htm) and [B3 Educação](https://edu.b3.com.br/w/opcoes-ibovespab3).
- **Style/liquidity:** Brazilian **equity** options are often **American-style** (early exercise possible) with liquidity concentrated in a handful of names and near-month series, whereas **Ibovespa index options (IBOV) are European-style and cash-settled** — assignment/early-exercise and liquidity caveats matter *more* on the equity side. Confirm contract specs per series.
- **Tools:**
  - **[opcoes.net.br](https://opcoes.net.br/)** — quotes, **historical & implied volatility** ([histórico de VI](https://opcoes.net.br/historico/volatilidade-implicita)), CALL/PUT chains, and a **Black-Scholes calculator** ([calculadora](https://opcoes.net.br/calculadora-Black-Scholes)), plus 3D implied-vol surfaces.
  - **[OpLab](https://oplab.com.br/)** — analysis/simulation, IV history, strategy builder, TradingView integration.
- *Lançamento coberto* (covered call) and *venda de put* (cash-secured put) are the most common income plays for Brazilian retail; same assignment/cost caveats apply.

---

## 9. A 6-Step Selection Checklist

1. **Form a directional thesis** (bull / bear / neutral / big-move) with a price target and timeframe.
2. **Check IVR / IV% and term structure** → decide buy vs. sell premium.
3. **Pick duration (DTE)** — long for buyers, ~30–60 for sellers, short only with strict gamma discipline.
4. **Choose defined vs. undefined risk** by capital, margin, and tolerance for tail loss.
5. **Compute breakevens, POP, and EV after costs** — reject thin-credit/poor-payoff trades.
6. **Pre-plan management:** profit target (~50%), stop/roll rules, and what assignment would mean.

---

## 10. Best Sources to Learn & Systematize

**Exchange / regulator education (free, authoritative):**
- The Options Industry Council (OIC) / OCC — interactive strategy guide by forecast & objective: [optionseducation.org](https://www.optionseducation.org/) · [The Options Strategies Quick Guide](https://www.optionseducation.org/the-options-strategies-quick-guide) · [Short (Iron) Condor strategy page](https://www.optionseducation.org/strategies/all-strategies/short-condor)
- Cboe Options Institute & research — SPX/0DTE and benchmark-index education: [0DTE SPX iron condor deep dive](https://www.cboe.com/insights/posts/henry-schwartzs-zero-day-spx-iron-condor-strategy-a-deep-dive/) · [BFLY (Iron Butterfly) & CNDR (Iron Condor) benchmark indices](https://www.cboe.com/insights/posts/benchmark-indices-series-volatility-management-with-cboes-bfly-and-cndr-indices/) · [Cboe 0DTE resources](https://www.cboe.com/tradable-products/0dte)
- Fidelity Options Strategy Guide — clear per-strategy mechanics: [Short Iron Condor Spread](https://www.fidelity.com/learning-center/investment-products/options/options-strategy-guide/short-iron-condor-spread)

**Practitioner / systematic (mechanics & data-driven rules):**
- tastylive / tastytrade — IVR-driven premium selling, 50%-profit management, volatility metrics: [Volatility Metrics (IVR, IV%, IVx, HV)](https://support.tastytrade.com/support/s/solutions/articles/43000539059) · [How can IV Rank be over 100 or below 0?](https://support.tastytrade.com/support/s/solutions/articles/43000559231)
- Option Alpha — POP-centric credit-spread playbooks & backtests: [Bull Put Spread](https://optionalpha.com/strategies/bull-put-credit-spread) · [SPY put-spread backtest](https://optionalpha.com/blog/spy-put-credit-spread-backtest)

**Books (the references serious traders cite):**
- Lawrence G. McMillan, *Options as a Strategic Investment* — [Amazon](https://www.amazon.com/Options-Strategic-Investment-Lawrence-McMillan/dp/0735201978)
- Sheldon Natenberg, *Option Volatility & Pricing* — [Amazon](https://www.amazon.com/Option-Volatility-Pricing-Strategies-Techniques/dp/0071818774)
- Euan Sinclair, *Option Trading: Pricing and Volatility Strategies and Techniques* — [Amazon](https://www.amazon.com/Option-Trading-Volatility-Strategies-Techniques/dp/0470497106)

**🇧🇷 Brazil:** [opcoes.net.br](https://opcoes.net.br/) · [OpLab guides](https://oplab.com.br/) · [B3 Educação](https://edu.b3.com.br/w/opcoes-ibovespab3)

---

> **Disclaimer:** This page is **educational/research material, not investment advice, a recommendation, or a solicitation**. Options involve substantial risk including total loss and (for naked/short positions) losses exceeding the premium received. Numbers are illustrative; verify current contract specs, margin, taxes, and costs with your broker/exchange. Past performance and backtests do not predict future results. **No options strategy is "free money."**

**Keywords:** options strategy selection, buy vs sell premium, implied volatility rank, IV percentile, probability of profit (POP), expected value, breakeven, theta decay, vega, volatility risk premium, iron condor, credit spread, covered call, cash-secured put, long straddle, IV crush, earnings event trading, term structure, contango, backwardation, undefined risk, assignment, early exercise; *opções, volatilidade implícita, venda de prêmio, compra de prêmio, decaimento temporal, probabilidade de lucro, ponto de equilíbrio, trava de alta/baixa, condor de ferro, lançamento coberto, venda de put, esmagamento de volatilidade, exercício/atribuição, B3, Ibovespa, OpLab*.
