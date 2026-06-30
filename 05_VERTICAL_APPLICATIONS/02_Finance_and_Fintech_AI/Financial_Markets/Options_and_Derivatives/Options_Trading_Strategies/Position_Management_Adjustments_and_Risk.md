# Options Position Management, Adjustments & Risk

> Managing the position after the fill — the Greeks, entry/exit rules, rolling, adjustments, assignment, and sizing — is what actually separates winning options traders from losing ones. This is education/research, **not investment advice**.

Most retail option education stops at "here are the 12 strategies." The edge — and the survival — lives in *management*: how you enter (price, liquidity, IV), how you exit (profit target, time stop, loss limit), how you defend a tested side, and above all how big you trade. This page is the practical, risk-honest reference for that second half. Audience is Brazil-heavy: English body with Portuguese terms in parentheses; B3 specifics noted where relevant. Numbers like "50% profit" and "21 DTE" are *rules of thumb* popularized by tastylive/Option Alpha, not laws of nature — backtest them on your own data before trusting them.

---

## 1. The Greeks in Management (os "gregos" na gestão)

You do not manage a P&L number; you manage a Greek exposure that *produces* the P&L. Track these per-position **and** at the portfolio level.

| Greek | What it measures | Sign for short premium (vendido) | Management implication |
|---|---|---|---|
| **Delta** (delta) | dPrice / dUnderlying — directional exposure | Starts ~neutral (strangle) or chosen bias | Hedge with stock/futures or by rolling the untested side to re-center |
| **Gamma** (gama) | dDelta / dUnderlying — how fast delta moves | **Short / negative** | The enemy near expiry; small moves swing delta hard → why you exit early |
| **Theta** (teta) | dPrice / dTime — time decay | **Positive** (you collect) | Your income; peaks the last weeks but so does gamma |
| **Vega** (vega) | dPrice / dIV — volatility exposure | **Short / negative** | You profit on IV drop (IV crush); a vol spike hurts before direction does |
| **Rho** (rô) | dPrice / dRates | Minor for short-dated | Matters for LEAPS / Brazil's high-rate (Selic) environment |

**Gamma/theta tradeoff (o trade-off central):** short-premium sellers are paid theta to carry short gamma. Far from expiry, theta is modest and gamma is tame — comfortable but slow. Near expiry, theta accelerates *but so does gamma*: a position that decayed peacefully for weeks can blow up in two days because delta now flips violently around the strike. This is the entire rationale for closing/rolling before the last weeks rather than "letting it expire for max profit."

**Second-order Greeks near expiry (charm / vanna / vomma):**
- **Charm** (delta decay over time) scales *inversely* with time: a 30-day option has modest charm, a 0DTE option has enormous charm. As expiry nears, your delta "melts" toward 0 or ±1 even with the underlying still — you must re-hedge just because of the calendar. ([macroption.com/second-order-greeks](https://www.macroption.com/second-order-greeks/))
- **Vanna** (dDelta/dIV = dVega/dUnderlying): an IV move silently changes your delta. In a selloff IV rises *and* price falls, so a "delta-neutral" book can become directional from vanna alone.
- **Vomma** (dVega/dIV): vega itself grows as IV rises, so a vol spike is *exponentially* worse than a linear vega estimate suggests — the reason short-vol drawdowns are fat-tailed.

> See: [Cboe — Learning the Greeks](https://www.cboe.com/insights/posts/learning-the-greeks-an-experts-perspective/) · [OIC — Volatility & the Greeks](https://www.optionseducation.org/advancedconcepts/volatility-the-greeks) · [Macroption — Second-Order Greeks](https://www.macroption.com/second-order-greeks/)

**Delta hedging (hedge de delta):** to neutralize directional risk, short the option's delta in the underlying (or futures). A short 30-delta strangle that drifts to net +25 delta can be flattened by selling ~25 shares-equivalent, or — preferred by premium sellers — by rolling the *untested* side toward the money to collect more credit while re-centering delta (no new capital, see §4). Re-hedging too often bleeds transaction costs; too rarely leaves you exposed. There is no free lunch — hedging converts directional risk into path-dependent cost.

---

## 2. Entry Rules (regras de entrada)

A position is half-decided at entry. Screen *before* you click.

| Filter | Practical threshold (rule of thumb) | Why it matters |
|---|---|---|
| **IV Rank / IV Percentile** (IV rank) | Sell premium when IV Rank is high (e.g. > 30–50); buy premium when low | You want to sell when options are "expensive," buy when "cheap." IV Rank normalizes IV vs its own 1-yr range |
| **Liquidity / volume** (liquidez) | Trade names with tight markets & real volume | Wide books mean you pay the spread twice (in + out) |
| **Bid-ask spread** (spread comprador-vendedor) | Penny-to-nickel on liquid underlyings; reject if spread is a large % of credit | Slippage is a guaranteed cost; a $0.10 spread on a $0.50 credit is a 20% tax |
| **Open interest** (contratos em aberto) | Prefer strikes with meaningful OI | Higher OI generally → better fills, easier rolls/exits |
| **DTE at entry** (dias até o vencimento) | ~30–45 DTE is the common short-premium sweet spot | Balances theta richness vs gamma risk |
| **Earnings / events** (resultados) | Know the date; IV crush post-earnings is a strategy, not a surprise | Selling into earnings is a bet on IV crush + range, not direction |

> B3 note: liquidity in Brazil concentrates in a handful of names. The most liquid **opções sobre ações** (single-stock options) cluster around **PETR4, VALE3, BBAS3, B3SA3, BOVA11**; **opções sobre Ibovespa** (index options, ticker family IBOV) serve broad-market views. Screen real two-sided markets on [opcoes.net.br](https://opcoes.net.br/opcoes/bovespa) or [OpLab](https://oplab.com.br/) — many strikes are illiquid and the displayed mid is fiction. Per [B3](https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/renda-variavel/opcoes-sobre-acoes.htm), single-stock options can be issued **European *or* American-style** ("Europeu ou Americano") — the liquid equity call series are American-style, so **early-exercise risk is real** and must be checked contract-by-contract; Ibovespa index options are **European-style**.

> Tools: [OIC Options Monitor](https://www.optionseducation.org/videolibrary/oic-options-monitor-tutorial) (delta, IV, bid-ask, volume) · [Cboe analytics](https://www.cboe.com/insights/posts/order-types-and-off-screen-liquidity-what-you-see-isnt-always-what-you-get/) on off-screen liquidity.

---

## 3. Exit Rules (regras de saída)

Exits beat entries for consistency. Pre-commit to all three triggers and act on whichever fires first.

| Exit trigger | Common rule (tastylive convention) | Rationale |
|---|---|---|
| **Profit target** (alvo de lucro) | Close short premium at **~50% of max profit** | tastylive studies show managing winners at 50% raises win rate and reduces tail risk vs holding to expiration; last 50% of credit carries disproportionate gamma risk |
| **Time stop** (stop de tempo) | Close/roll at **~21 DTE** regardless of P&L | Gamma risk accelerates in the final ~3 weeks; the favorable theta/gamma ratio deteriorates |
| **Loss limit** (limite de perda) | e.g. close at **2x credit received** loss, or at a fixed % of buying power | Caps left-tail damage; mechanical, not emotional |

The platform mechanic: a "close at profit percent" order computes the closing price from the credit received — sell a put for \$1.00 credit, a 50%-of-max-profit close lists a \$0.50 *debit* to buy it back. ([tastytrade support — Close at Profit Percent](https://support.tastytrade.com/support/s/solutions/articles/43000435423))

**Honest caveats:** 50%/21-DTE are *defaults*, not universal optima — they were derived mostly on liquid US index/ETF underlyings (SPX/SPY) over specific regimes. On Brazilian single names with wider spreads and different vol dynamics, re-test. A profit target also caps upside; in a strong trend you'll leave money behind — that's the deliberate price of a higher win rate.

> Sources: [tastylive Learn Center (free courses)](https://tastylive.freshdesk.com/support/solutions) · [tastytrade — Close at Profit Percent order](https://support.tastytrade.com/support/s/solutions/articles/43000435423)

---

## 4. Rolling (rolagem)

Rolling = close the current option and open another, ideally **for a net credit**, to buy time or re-center risk. It is not a magic loss-eraser; rolling a deeply tested directional position can *compound* the loss by adding duration to a wrong bet.

| Roll type | Mechanic | When | Watch-out |
|---|---|---|---|
| **Roll out** (rolar no tempo / duração) | Same strike, later expiry | Need more time for thesis; collect extra credit | Extends exposure; only do it for a credit |
| **Roll up / down** (rolar para cima/baixo) | Move strike with the underlying | Re-center delta after a move | Moving the *tested* side usually compounds loss — avoid |
| **Roll the untested side** (rolar o lado não testado) | Bring the safe leg toward the money | Strangle/condor gets directional | The tasty/Option Alpha core defense: take credit, widen breakeven on the threatened side |
| **Roll for duration + credit** | Out in time *and* collect net credit | 21-DTE management on a still-valid thesis | If you can't roll for a credit, the thesis may be broken — consider closing |

**The single most important adjustment rule (Option Alpha):** *do not roll the challenged/tested side* — it compounds the loss. Instead, roll the **untested** side toward the money. On a strangle that's being tested to the upside, you roll the *put* side up: you collect more credit, narrow the structure, and widen the breakeven on the call (challenged) side, improving probability of profit without paying a debit. Option Alpha's stated principle: **never pay a debit to adjust — always take in net credit.** ([Option Alpha — Trade Adjustments course](https://optionalpha.com/courses/trade-adjustments))

> Sources: [Option Alpha — Trade Adjustments (free course)](https://optionalpha.com/courses/trade-adjustments) · [Option Alpha — Option Adjustment Principles (podcast)](https://optionalpha.com/podcast/option-adjustment-principles)

---

## 5. Adjustments — Situation → Response (ajustes: situação → resposta)

Adjustments add legs to morph the position's risk. Each reduces one risk by adding another; none are free.

| Situation | Adjustment | Effect | Cost / risk |
|---|---|---|---|
| Short strangle, one side tested | **Roll untested side toward ATM** | More credit, re-centered delta, wider breakeven on tested side | Tightens the structure; an opposite reversal can now test the *other* side |
| Naked short call/put running against you | **Convert to a spread** (buy a further wing) | Caps the undefined tail; defines max loss | Pays a debit; locks in some loss |
| Iron condor, market grinds toward a short strike | **Roll the untested vertical in** | Net credit, narrower condor | Less room to the other side |
| Tested side blown through, want to stay | **Go "inverted" strangle** | Invert strikes for extra credit, bet on mean reversion | Caps max profit below full credit; pure reversion bet |
| Want to reduce naked risk pre-emptively | **Add wings → iron condor/iron fly** | Defined risk, lower buying-power reduction | Caps profit; pays for protection |
| Directional winner, lock gains | **Roll up-and-out (debit spread) / convert to fly** | Bank profit, keep cheap upside | Reduces remaining theta/credit |

> **Brutal honesty:** "adjusting" is often a polite name for *averaging into a losing trade*. Each leg adds commissions, slippage, and sometimes risk on the *other* side. If the original thesis is broken, the cleanest adjustment is **closing**. Adjust to manage probability on a still-valid thesis — not to avoid admitting a loss.

> Sources: [Option Alpha — Iron Condor Adjustments](https://optionalpha.com/lessons/iron-condor-adjustments) · [Option Alpha — Strangle/Straddle Adjustments](https://optionalpha.com/lessons/straddle-adjustments) · McMillan, *Options as a Strategic Investment*, 5th ed. (adjustment & follow-up chapters).

---

## 6. Assignment & Early Exercise (exercício e atribuição)

| Risk | Trigger | Defense |
|---|---|---|
| **Early assignment** (atribuição antecipada) | American-style short option goes ITM; writer can be assigned *any time* | Monitor extrinsic value; close/roll ITM shorts before it vanishes |
| **Ex-dividend call assignment** (atribuição por dividendo) | Short **ITM call**, extrinsic value < upcoming dividend → holder exercises to capture the dividend | The #1 cause of early call assignment; close/roll ITM short calls *before* the ex-dividend date |
| **Pin risk** (risco de pin) | Underlying settles *right at* a short strike at expiry | You don't know if/how much you're assigned until after the close → unhedged weekend/overnight exposure. Close near-the-money shorts before expiry |
| **Exercise-by-exception** | OCC auto-exercises options **≥ \$0.01 ITM** at expiry unless instructed otherwise | Don't assume a barely-ITM long expires worthless; don't assume a short won't be assigned |

OCC assigns exercise notices via a **random** procedure across clearing accounts, so being short ITM is a probabilistic, not guaranteed, assignment — but plan as if it will happen. Assignment risk rises as an option goes deeper ITM and as expiry nears (less time premium to protect you).

> B3 note: single-stock **opções sobre ações** can be American or European-style (the liquid call series are American) → ex-dividend early-exercise and pin risk are live concerns on the American series, especially around the heavy Brazilian dividend calendar (PETR4, BBAS3, etc.). **Opções sobre Ibovespa são europeias** → no early exercise, but cash-settlement pin risk at the settlement print still applies.

> Sources: [OIC — Options Assignment FAQ](https://www.optionseducation.org/referencelibrary/faq/options-assignment) · [tastytrade — Early Assignment](https://support.tastytrade.com/support/s/solutions/articles/43000505597) · [Schwab — Risks of Options Assignment](https://www.schwab.com/learn/story/risks-options-assignment)

---

## 7. Position Sizing — the #1 Risk Control (dimensionamento de posição)

> No adjustment, no Greek, no exit rule matters if a single trade can ruin you. **Sizing is the dominant risk control.** A great strategy sized wrong is a blow-up waiting to happen.

| Method | Idea | Practical note |
|---|---|---|
| **Fixed fractional / % of BP** (fração fixa) | Risk a small fixed % of account or buying power per trade | tastylive convention: keep total premium/BP usage modest (e.g. ~25–50% of net liq deployed, far less when uncertain) |
| **Kelly criterion** (critério de Kelly) | Bet a fraction proportional to edge/odds | **Use fractional Kelly (¼–½)** — full Kelly is too aggressive once you account for parameter *misestimation*, return skew, and stops (Sinclair, *Volatility Trading*, Ch. 9 on sizing & Kelly) |
| **Buying-power reduction (BPR)** (redução de poder de compra) | The capital the broker freezes per trade | Defined-risk structures (spreads/condors) cut BPR vs naked shorts → more, smaller, uncorrelated trades |
| **Per-position max loss cap** | Cap notional loss per trade as % of net liq | Survive the tail; live to compound |

**Kelly honesty:** Kelly maximizes long-run growth *only if you know the true probabilities* — which in options you do **not**. Sinclair's treatment explicitly incorporates estimation uncertainty, skew, and stop-losses, and the practical conclusion is to bet a *fraction* of Kelly. Overbetting doesn't just lower returns — past a point it drives terminal wealth to zero.

> Sources: Euan Sinclair, [*Volatility Trading*, 2nd ed. (Wiley)](https://www.wiley.com/en-us/Volatility+Trading,+++Website,+2nd+Edition-p-9781118416723) — bet sizing & Kelly chapters · Euan Sinclair, [*Positional Option Trading* (Wiley)](https://www.amazon.com/Positional-Option-Trading-Wiley/dp/1119583519) — trade sizing & unknowable risk.

---

## 8. Portfolio-Level Greeks, Margin & Correlation (gregos e margem no nível do portfólio)

Per-trade discipline is undone if your *book* is secretly one big bet.

- **Aggregate the Greeks:** sum delta/gamma/vega/theta across positions. Many "diversified" books are net-short-vega and net-short-gamma — i.e., one short-vol position wearing ten tickers. A single vol spike hits all of them at once.
- **Correlation (correlação):** "uncorrelated" names correlate to ~1 in a crash. Beta-weight portfolio delta to a benchmark (SPX, or **Ibovespa/IBOV** for Brazil) to see your true directional exposure.
- **Reg-T vs Portfolio Margin (PM):** Reg-T charges margin per position (~50% initial on stock). PM evaluates *total* portfolio risk and can sharply cut requirements for hedged books (broker examples cite materially lower BPR; tastytrade quotes up to ~6.7:1 leverage vs ~2:1 under Reg-T). PM is more capital-efficient **and more dangerous** — lower margin means easier overleveraging, and in stress, historically uncorrelated legs can move together and negate the hedge the low margin assumed.

> Sources: [Schwab — How Portfolio Margin Works](https://www.schwab.com/learn/story/option-traders-how-portfolio-margin-works) · [tastytrade — What is Portfolio Margin](https://tastytrade.com/learn/accounts/account-resources/what-is-portfolio-margin-how-it-works/)

---

## 9. Tail-Risk Hedging (hedge de cauda)

Short-premium books make small money often and risk large losses rarely — the payoff is short the tail. Options to defend the tail:

- **Long OTM puts / put spreads** (puts/travas de proteção) on the index — a standing cost (theta bleed) that pays off in crashes; sizing the bleed vs protection is the hard part.
- **Reduce net short gamma/vega before known events** — cut size or buy back shorts ahead of CPI, FOMC, Copom (Brazil's rate decision), elections.
- **Keep dry powder** — cash/BPR headroom is itself a tail hedge; it lets you survive margin expansion and add when IV is richest.
- **Diversify *vega* sources**, not just tickers — own some long-vol/long-gamma so the book isn't a single short-vol position.

> **There is no free lunch.** Tail hedges cost carry in calm markets; that drag is the premium for not being wiped out. Backtest the *net* of hedge cost + strategy, not the strategy alone.

---

## 10. Systematizing It (sistematização)

Codify your entry/exit/management rules and **backtest before trusting any "50%/21-DTE" number**:

- **[optopsy](https://github.com/goldspanlabs/optopsy)** — Python options backtesting library (now maintained by Goldspan Labs; the older `michaelchu/optopsy` URL redirects here); supports profit-target, stop-loss, and **DTE-based exits** (`exit_dte`, `max_entry_dte`) and built-in multi-leg strategies — ideal for testing management rules on historical chains.
- **[py_vollib](https://github.com/vollib/py_vollib)** / **[QuantLib](https://www.quantlib.org/)** — pricing, implied vol, and Greeks computation.
- **[awesome-options-analytics](https://github.com/FlashAlpha-lab/awesome-options-analytics)** — curated list of options tools, APIs, papers, and education.
- **B3 / Brazil:** [OpLab](https://oplab.com.br/) (30+ pre-configured strategies, vol/volume heatmaps, custody sync) and [opcoes.net.br](https://opcoes.net.br/) (free chains, studies, screeners).

---

## Honest Disclaimers (avisos honestos)

- **Not investment advice.** This is educational/research material. Options can lose 100%+ of capital (undefined-risk short positions can lose *far* more than the premium received).
- **No strategy is "free money."** High-win-rate premium selling trades frequent small gains for rare large losses; the expectancy is not magically positive.
- **Adjustments can compound losses** — they are probability management on a valid thesis, not loss erasers.
- **Rules of thumb (50%, 21 DTE, 2x stop, fractional Kelly)** are regime- and market-dependent; backtest on *your* instruments (incl. B3 liquidity/spreads) before relying on them.
- **Costs are real and recurring:** spreads, commissions, slippage, financing, and taxes erode edge — model them.

---

**Sources:** [tastytrade — Close at Profit Percent](https://support.tastytrade.com/support/s/solutions/articles/43000435423) · [tastylive Learn Center](https://tastylive.freshdesk.com/support/solutions) · [tastytrade — Early Assignment](https://support.tastytrade.com/support/s/solutions/articles/43000505597) · [tastytrade — Portfolio Margin](https://tastytrade.com/learn/accounts/account-resources/what-is-portfolio-margin-how-it-works/) · [Option Alpha — Trade Adjustments course](https://optionalpha.com/courses/trade-adjustments) · [Option Alpha — Adjustment Principles](https://optionalpha.com/podcast/option-adjustment-principles) · [Option Alpha — Iron Condor Adjustments](https://optionalpha.com/lessons/iron-condor-adjustments) · [Cboe — Learning the Greeks](https://www.cboe.com/insights/posts/learning-the-greeks-an-experts-perspective/) · [OIC — Volatility & the Greeks](https://www.optionseducation.org/advancedconcepts/volatility-the-greeks) · [OIC — Options Assignment FAQ](https://www.optionseducation.org/referencelibrary/faq/options-assignment) · [Macroption — Second-Order Greeks](https://www.macroption.com/second-order-greeks/) · [Schwab — Portfolio Margin](https://www.schwab.com/learn/story/option-traders-how-portfolio-margin-works) · [Schwab — Assignment Risks](https://www.schwab.com/learn/story/risks-options-assignment) · Euan Sinclair, [*Volatility Trading* 2nd ed.](https://www.wiley.com/en-us/Volatility+Trading,+++Website,+2nd+Edition-p-9781118416723) · Euan Sinclair, [*Positional Option Trading*](https://www.amazon.com/Positional-Option-Trading-Wiley/dp/1119583519) · Lawrence McMillan, [*Options as a Strategic Investment*, 5th ed.](https://www.penguinrandomhouse.com/books/310812/options-as-a-strategic-investment-by-lawrence-g-mcmillan/) · [optopsy (GitHub)](https://github.com/goldspanlabs/optopsy) · [OpLab](https://oplab.com.br/) · [opcoes.net.br](https://opcoes.net.br/) · [B3 — Opções sobre Ações](https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/renda-variavel/opcoes-sobre-acoes.htm)

**Keywords:** options position management, adjustments, rolling, defending the tested side (rolagem, defesa do lado testado), the Greeks (delta, gamma, theta, vega, charm, vanna, vomma / gregos), IV rank, profit target 50%, 21 DTE time stop, loss limit, early exercise, assignment, ex-dividend, pin risk (exercício antecipado, atribuição, risco de pin), position sizing, Kelly criterion, buying power reduction, portfolio margin (dimensionamento, critério de Kelly, redução de poder de compra, margem de portfólio), tail-risk hedging (hedge de cauda), B3, opções sobre ações, opções sobre Ibovespa, OpLab, opcoes.net.br, tastylive, Option Alpha, McMillan, Sinclair, Cboe, OIC, optopsy
