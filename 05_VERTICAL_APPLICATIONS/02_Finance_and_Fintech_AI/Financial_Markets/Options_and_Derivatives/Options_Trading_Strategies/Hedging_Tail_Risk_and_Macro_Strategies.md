# Macro: Hedging, Tail Risk & Macro-Driven Options

> Portfolio-level reference for hedging, tail-risk convexity, VIX/volatility-regime trades, and macro-event positioning — payoffs, breakevens, Greeks, and honest cost/decay warnings, with verified sources (global, 2024–2026). The MACRO companion to the single-trade sibling pages. Brazil-aware: English body with Portuguese terms and B3 specifics. **Not investment advice — research/education only (*material educacional, não é recomendação de investimento*).**

This page is deliberately the *portfolio/macro* layer. For single-trade construction (verticals, straddles, ratios, PMCC, condors, calendars) see [Directional & Spread](./Directional_and_Spread_Strategies.md) and [Volatility, Income & Neutral](./Volatility_Income_and_Neutral_Strategies.md). For Greeks-based adjustment of an existing book see [Position Management](./Position_Management_Adjustments_and_Risk.md).

---

## 1. The hedging dilemma — there is no free hedge

A hedge that always pays in a crash has **negative expected return** in the 90%+ of months that aren't crashes (you pay carry/theta). A hedge that's cheap to carry usually fails to pay when you need it. Every hedging decision trades **convexity vs. carry**.

- **Sizing dominates structure.** *How much* you hedge (notional, % NAV/year budget) drives outcomes far more than *which* structure you pick. A perfectly chosen hedge sized at 0.1% of NAV does little; a clumsy one sized correctly can save a portfolio.
- **The volatility tax (drag of losses).** Geometric (compounded) return ≈ arithmetic return − ½·variance. Large drawdowns hurt compounding non-linearly, which is the economic case *for* tail mitigation — but only if the hedge's carry cost is less than the compounding benefit it buys. Framing per Spitznagel's *Safe Haven* and AQR's tail-hedging work (see below).

> **Honest note:** "cost-effective tail hedging" is genuinely hard. Most static long-put programs lose money over full cycles ([CBOE PPUT](https://www.cboe.com/us/indices/dashboard/pput/) drag — see §2). Convexity is paid for; the question is whether it's paid for *cheaply enough*.

---

## 2. Portfolio hedging menu (with breakevens & honest warnings)

For a long equity book. Costs are qualitative (regime-dependent); confirm live.

| Hedge | Structure | Carry cost | Upside given up | Breakeven / what it protects | Honest warning |
|---|---|---|---|---|---|
| **Protective put** (*put protetora*) | Long index put vs. portfolio | High | None | Floors NAV below strike; BE = price paid back via decline below strike + premium | Bleeds in calm markets; rolling rich IV is expensive |
| **Put spread** (*trava de baixa*) | Buy put, sell lower put | Medium | None | Protects a **band** only (long − short strike); BE = long strike − net debit | **Not crash insurance** — protection stops at short strike; a deeper crash is unhedged below it |
| **Collar** (*collar*) | Long put + short call vs. shares | Low / ~zero | **Yes** (capped at call) | Floors below put, caps above call | Caps upside; a strong rally is forgone — opportunity cost is the real cost |
| **Put-spread collar** | Put spread financed by short call | Very low | Yes | Banded floor + capped upside | Combines both limitations: capped upside *and* a floor that gives out below the short put |
| **Ratio put spread** | Buy 1 put, sell 2+ lower puts | Credit/cheap | None | — | **Net short tail below the short strikes — this is NOT a hedge.** It *adds* crash risk for a credit. Listed only to warn |
| **Index/futures overlay** | Short ES/MES or index puts beta-weighted to book | Varies | Yes (futures) | Linear short delta offsets beta | Basis risk; futures short has symmetric P/L (no convexity) |

**CBOE PPUT reality check.** The [Cboe S&P 500 5% Put Protection Index (PPUT)](https://www.cboe.com/us/indices/dashboard/pput/) holds the S&P 500 plus a **monthly 5%-OTM SPX put**, rolled on the third Friday ([methodology PDF](https://cdn.cboe.com/api/global/us_indices/governance/Cboe_SP_500_Put_Protection_Indices_Methodology.pdf)). The first 5% of any monthly drawdown is **unhedged** (full market exposure); only losses beyond 5% within the cycle are mitigated — and the continuous premium is a long-run drag. Static put-buying underperforms in most regimes; that's the core tail-hedging problem, not a quirk.

OIC mechanics: [Protective Put](https://www.optionseducation.org/strategies/all-strategies/protective-put-married-put), [Collar](https://www.optionseducation.org/strategies/all-strategies/collar-protective-collar).

---

## 3. Index vs. single-name hedging & beta-weighting the delta

Hedge **systematic (market) risk** with a broad index; idiosyncratic single-name risk needs name-specific hedges. To size an index hedge, **beta-weight** every position's delta to one benchmark so the whole book reads as "≈ X benchmark-shares long," then short that delta ([tastytrade — beta-weighted deltas](https://support.tastytrade.com/support/s/solutions/articles/43000522492), [Cboe — right-size hedges with XSP](https://www.cboe.com/insights/posts/how-to-right-size-hedges-via-beta-weighting-with-xsp-options/)).

**US delta/notional building blocks (confirm live):**

| Instrument | Multiplier / size | Settlement | Note |
|---|---|---|---|
| **SPX** options | $100 × index | European, **cash** | No early assignment; ~10× a SPY/XSP notional |
| **XSP** (Mini-SPX) options | $100 × (index/10) | European, cash | 1/10 SPX; same notional as SPY options; good for precise sizing |
| **SPY** options | 100 × ETF px (≈ index/10) | American, physical | Liquid; early-assignment & dividend risk |
| **ES** (E-mini) futures | $50 × index | Futures | 1 pt = $50 |
| **MES** (Micro E-mini) | $5 × index | Futures | 1 pt = $5; fine-grained futures hedge |

**🇧🇷 B3 analogues:** index options on **Ibovespa** ([B3 — Opções sobre Ibovespa](https://edu.b3.com.br/w/opcoes-ibovespab3)); puts on the Ibovespa ETF **BOVA11**; and futures overlays via **mini-índice WIN** (R$0,20/ponto) or full **IND** (R$1,00/ponto) ([B3 — Mini-índice](https://edu.b3.com.br/w/mini-indice)). Liquidity is concentrated; far-dated index hedges are scarce vs. the US.

**Trade-off:** index hedges are cheap/liquid but leave **basis risk** (your book ≠ the index); single-name hedges kill basis risk but cost more and are illiquid OTM.

---

## 4. Tail hedging & convexity (deep-OTM, ladders, VIX overlays)

Tail hedges aim for **convexity**: tiny carry, huge non-linear payoff in a fast crash.

- **Deep-OTM index puts / put ladders** — buy small, far-OTM puts (or a ladder across strikes/expiries). **~90–95% expire worthless**; the rare winners are paid for by all the losers, so the program only works if the gap-down is violent enough that gamma/vega explode. Size as a **budget (~0.5–1% NAV/year is a common framing)**, not a position.
- **VIX-call overlay (VXTH-style).** The [Cboe VIX Tail Hedge Index (VXTH)](https://cdn.cboe.com/api/global/us_indices/governance/Cboe_VIX_Tail_Hedge_Index_Methodology.pdf) holds the S&P 500 and overlays **monthly 30-delta VIX calls**, with the call weight scaled by VIX: **0%→1% as VIX runs 15→30, ½% from 30→50, and nothing bought outside the 15–50 corridor.** In Oct 2008 the overlay gained while the S&P 500 fell sharply — convexity at work — but it costs carry in calm regimes.
- **VIX call spreads** cap both cost and payoff — cheaper carry, but they **cap your convexity** exactly when an uncapped tail hedge would pay most.

**Two doctrines, honestly contrasted:**
- **Spitznagel / Universa — *Safe Haven* (Wiley, 2021, ISBN 9781119401797):** explosive, deep-OTM convexity sized small; judge a hedge by its effect on the *whole portfolio's compound growth*, not its standalone return. [Wiley](https://www.wiley.com/en-us/Safe+Haven:+Investing+for+Financial+Storms-p-9781119401797).
- **AQR — *Tail Risk Hedging: Contrasting Put and Trend Strategies* (July 2020):** systematic put-buying gives reliable but expensive crash protection with negative long-run return; **trend-following** gives less reliable, slower protection but **positive** long-run return — different crash shapes. [AQR white paper (PDF)](https://images.aqr.com/-/media/AQR/Documents/Insights/White-Papers/AQR-Tail-Risk-Hedging-Contrasting-Put-and-Trend-Strategies.pdf).

> Reconcile, don't pick a winner: puts/VIX = **fast-gap** insurance (convex, costly); trend = **slow-grind** insurance (cheap-ish, positive carry, lags a one-day crash). They protect different disasters.

---

## 5. VIX-based hedging & the term-structure trap

VIX is **not directly tradable**; you access it via VIX futures/options or volatility ETPs. The structural trap:

- **VIX futures are in contango ~75–80%+ of days** (front < back), so long-vol ETPs **roll up the curve and bleed**. [VIX Central](https://vixcentral.com/) / [VIXStructure](https://vixstructure.com/) visualize this live; [quantvps — VIX futures curve explained](https://www.quantvps.com/blog/vix-futures-curve-explained).
- **VXX / UVXY are wealth incinerators for buy-and-hold.** Since inception (VXX: Jan 2009) VXX is down **~98–99%+** from roll decay; leveraged UVXY is worse. Use only as **short-dated** tactical hedges, never as a carry position. [tastytrade — what is VXX](https://tastytrade.com/learn/trading-products/stocks/what-is-VXX-how-to-trade-it/).
- **Prefer defined-risk VIX *calls* / call spreads** over holding VXX/UVXY when you want a spike hedge — your loss is the premium, not open-ended decay.
- **Backwardation (front > back) is itself a stress signal** — the curve usually inverts only in/after a vol shock, by which point the cheap-hedge window has closed.

> The term-structure trap means *timing and instrument choice* dominate: the same VIX view expressed via a held ETP vs. a defined-risk call spread can be the difference between bleeding out and a clean hedge.

---

## 6. Volatility-regime strategies (VRP & regime detection)

The **variance risk premium (VRP)** — implied vol tends to exceed subsequently realized vol — pays option *sellers* on average, but it is **compensation for bearing tail/gap risk, not free money**, and it has compressed in modern markets. [Quantpedia — VRP effect](https://quantpedia.com/strategies/volatility-risk-premium-effect).

| Regime (rough) | Signal | Lean |
|---|---|---|
| **Low VIX + steep contango** | VIX low, curve upward | Cautious **long-vol/convexity** is cheap; short-vol carry is thin per unit of tail risk |
| **High VIX + backwardation** | VIX elevated, curve inverted | Short-vol *can* pay (rich premium) but you're selling into a steamroller — defined risk only |
| **Transition** | IV Rank mid; curve flattening | Reduce size; regime is unstable |

**Regime detection inputs:** VIX *level* + **term-structure slope** (M1:M2) + **IV Rank/Percentile**. None is sufficient alone. See the [Systematic & ML page](./Systematic_and_ML_Options_Strategies.md) for harvesting VRP systematically and its drawdowns; academic context: [Iyer (2025), *Shorting Volatility: Harvesting the Risk Premium and Managing Tail Risk*, SSRN 5464595](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5464595) (notes inverse-vol notes that returned ~40%/yr then **collapsed >90% in Feb-2018 "Volmageddon"**).

---

## 7. Macro-event positioning & IV crush

Implied vol **inflates into** scheduled macro events, then **crushes** the instant uncertainty resolves — so a correct directional view can still *lose* if you were long premium into the crush. [SpotGamma — IV crush](https://support.spotgamma.com/hc/en-us/articles/15249330755859-IV-Crush-Explained-What-It-Is-When-It-Happens-and-How-to-Trade-It).

| Event | Cadence | Typical vol behavior |
|---|---|---|
| **FOMC** decision | 8×/yr | IV builds in; sharp post-statement crush |
| **CPI / PCE** | monthly | Macro-IV cycle mirrors earnings |
| **NFP** (payrolls) | monthly | Front-week / 0DTE IV elevated, fast crush |
| **Elections / referenda** | episodic | Long-dated event premium, multi-week |
| **Single-name earnings** | quarterly | Classic pre-event IV ramp → post-event crush |

- **Implied (expected) move ≈ ATM straddle × ~0.85** (≈ ±1σ for the expiry). It tells you what the market has *already priced*; beating it requires the realized move to exceed the priced move **net of the IV crush**.
- **Use defined-risk** around binary events. Long premium needs a move bigger than priced; short premium harvests the crush but carries gap risk. 0DTE around FOMC/CPI is *especially* gamma-dangerous.

---

## 8. Correlation / dispersion as a macro-vol trade

Index vol is cheaper than the average of its constituents' vol because diversification suppresses index moves — the gap is the **correlation risk premium**. The classic **dispersion** trade is **short index vol / long single-name vol** (sell the index straddle, buy constituent straddles), profiting when correlation falls (names move idiosyncratically).

- Forward-looking gauge: the [Cboe S&P 500 Dispersion Index (DSPX)](https://www.cboe.com/us/indices/dispersion/) (launched Sep 2023) measures expected 30-day dispersion via a VIX-style calc on index + single-stock options.
- **Honest risk:** classic dispersion is **short correlation = short tail**. In a systemic crash correlations spike toward 1, the short index-vol leg blows out, and the long single-name legs don't compensate. It is a crisis-fragile carry trade, not a hedge. [Quantpedia — Dispersion trading](https://quantpedia.com/strategies/dispersion-trading).

---

## 9. Crisis alpha — match the hedge to the crash *shape*

No single hedge wins every crash; diversify hedges by **how the crash unfolds**.

| Crash shape | Example | Pays well | Pays poorly |
|---|---|---|---|
| **Fast gap / overnight** | Mar 2020 COVID, Aug-2024 vol spike | Long puts, long VIX calls (convexity) | Trend-following (no time to flip) |
| **Slow grind / regime bear** | 2000–02, 2022 | Trend-following / managed futures | Static puts (theta bleeds over the long decline) |
| **Vol-of-vol blowup** | Feb 2018 Volmageddon | Long-vol, owning convexity | Short-vol / inverse-VIX (catastrophic) |

> **Diversify the *insurer*, not just the *insured*.** Combine fast-gap convexity (puts/VIX) with slow-grind trend exposure so you're covered across crash shapes — at the cost of carrying both. Per AQR's put-vs-trend framing (§4).

---

## 10. Disciplined hedging checklist + honesty note

1. **Budget first.** Set a max annual hedge spend (% NAV) and size *to it*; never let a tail program quietly grow.
2. **Name the disaster.** Fast gap? Slow grind? Single-name? Pick the structure that pays *that* shape.
3. **Beta-weight the book** to one benchmark before sizing any index hedge.
4. **Respect the term-structure trap.** Don't hold VXX/UVXY for carry; prefer defined-risk VIX calls/spreads.
5. **Mind IV regime.** Buying protection is dearest exactly when fear is already priced; layer in calm.
6. **Defined risk around macro events;** assume the IV crush.
7. **Backtest with real costs** (slippage, commissions, roll, assignment) — see [Backtesting & Frameworks](../../Backtesting_and_Frameworks/).
8. **Survival over optimization.** A hedge that lets you stay solvent through the worst path beats one with a prettier average.

### 🇧🇷 B3 / Brazil tax & regulatory notes (confirm current)
- **Options & derivatives are taxed on monthly net gains: 15% (operações normais) / 20% (day trade)**, with a small *dedo-duro* withholding (≈1% day-trade / 0.005% normal) — credited against the due tax, which you compute and pay via **DARF (código 6015)** monthly.
- **The R$20.000/month sales-exemption applies ONLY to spot-stock (mercado à vista) sales — it does NOT apply to options or other derivatives.** A common, costly myth. ([B3 — declarar derivativos](https://borainvestir.b3.com.br/noticias/imposto-de-renda/renda-variavel-imposto-de-renda/de-opcoes-a-contratos-futuros-como-declarar-derivativos-no-imposto-de-renda-sem-cair-na-malha-fina/), [InfoMoney — opções no IR](https://www.infomoney.com.br/guias/opcoes-de-acoes-imposto-de-renda-ir/)).
- These rates are **still current for 2026**: the unifying-rate proposal (MP 1.303/2025, which would have moved to a single ~17,5% rate and quarterly apuração) **lost validity on 8 Oct 2025** when the Câmara pulled it from the agenda, so the legacy 15%/20% regime persists. Still, confirm with a contador and current CVM/Receita guidance before trading — tax rules change.

> **Risk note (read twice).** Hedging is not free and most static hedges lose money over full cycles; convexity is *paid for*. Long options can lose 100% of premium; **short-vol / ratio / dispersion structures can lose multiples of the credit in a crash.** Volatility ETPs (VXX/UVXY) decay relentlessly. This is research/education, **not investment advice** (*não é recomendação de investimento*). Backtest with realistic costs, paper-trade, size for survival, and confirm every mechanic with your broker, the OIC/CBOE, and B3.

**Keywords:** hedging, tail risk (risco de cauda), tail hedging, convexity (convexidade), volatility tax, protective put (put protetora), put spread (trava de baixa), collar, put-spread collar, ratio put spread, index overlay, beta-weighted delta (delta beta-ponderado), SPX, XSP, SPY, ES, MES, VIX, VXTH (VIX Tail Hedge), VXX, UVXY, contango, backwardation, term structure (estrutura a termo), variance risk premium (prêmio de risco de variância, VRP), volatility regime, IV rank, IV crush (esmagamento de IV), expected/implied move, FOMC, CPI, PCE, NFP, dispersion (dispersão), correlation risk premium, DSPX, crisis alpha, trend following, Spitznagel Safe Haven, AQR put vs trend, Universa, B3, Ibovespa, BOVA11, mini-índice WIN, IND, imposto de renda derivativos, DARF, CVM, OIC, CBOE, opcoes.net.br, OpLab
