# Betting Exchanges, Trading & Market Microstructure

> Authoritative, worldwide reference on **betting exchanges (bolsas de apostas)** as *markets*: how peer-to-peer back/lay pricing works, the developer **APIs & open-source trading frameworks** (Betfair, Smarkets, Matchbook, Betdaq, ProphetX), the vocabulary of **sports trading** (scalping, swing, greening-up/hedging, in-play), the **closing line as consensus**, and the deep **parallels to financial-market microstructure** (order book, maker/taker, liquidity, adverse selection). New angle for this section — complements the modelling/odds/OSS pages. **Research & education only (pesquisa e educação — data science / market microstructure), current 2024–2026.**

> ⚠️ **NOT betting or investment advice (NÃO é dica de aposta nem recomendação financeira).** An exchange is not a money machine — it is a **near-efficient market** where you trade against other sharp participants and pay commission on every winning position. On deep Betfair markets the **closing/Betfair Starting Price is the wisdom-of-the-crowd consensus** and is very hard to beat; "trading" losses, spread and commission compound. **Only ~2–5% of bettors are profitable long-term; most lose.** Nothing here is a strategy, tip, or system. If gambling stops being fun or you cannot stop, get help now: [BeGambleAware](https://www.begambleaware.org/) · [GamCare](https://www.gamcare.org.uk/) · [Gambling Therapy](https://www.gamblingtherapy.org/) (multilingual) · [Gamblers Anonymous](https://www.gamblersanonymous.org/) · 🇧🇷 [Jogo Responsável — SPA/MF](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas) · 🇧🇷 CVV **188** · [cvv.org.br](https://www.cvv.org.br/).

**Sibling pages (do not duplicate):** models → [Innovative Models & Deep Learning](./Innovative_Models_and_Deep_Learning.md) · data → [Global Datasets & APIs](./Global_Datasets_and_Data_APIs.md) · tools → [Open-Source Tools & Libraries](./Open_Source_Tools_and_Libraries.md) · odds/value/**CLV**/staking → [Bet Selection, Staking & High-Odds Analysis](./Bet_Selection_Staking_and_High_Odds_Analysis.md) · [Kaggle Datasets](./Kaggle_Football_Datasets_and_Competitions.md).

---

## 1) Exchange vs bookmaker: two different market structures

A **bookmaker (casa de apostas / *quote-driven / dealer market*)** quotes prices, takes the other side of your bet, and builds in a margin (**overround / vigorish**); it profits by balancing its book and by limiting winners. A **betting exchange (bolsa de apostas / *order-driven / auction market*)** is a **peer-to-peer** venue: it matches customers against each other and never takes a position. You can **back (a favor)** an outcome (bet it will happen — you are the *taker/backer*) **or lay (contra)** it (bet it will *not* happen — you act like the bookmaker). The exchange earns a **commission (comissão)** on net winnings, not a spread.

| Feature | Bookmaker (dealer) | Exchange (auction / P2P) |
|---|---|---|
| Who is the counterparty | The house | Another customer |
| Price mechanism | House posts a marked-up quote | **Limit order book**: best available back/lay from the crowd |
| Margin model | **Overround** baked into odds (typ. 3–8%) | ~0% margin in the odds; **commission on net winnings** (~2–5%) |
| Can you *lay* (sell)? | No | **Yes** (core feature) |
| Limits winners? | Yes — restricts/bans (*gubbing*) | Rarely (P2P) — but pays **premium/expert-fee** on large sustained profit |
| Efficiency | Slower, softer lines | Order-driven markets are **more efficient** ([research](https://www.sciencedirect.com/science/article/abs/pii/S0169207021000996)) |

Economically the exchange is a **prediction market**: the traded price ≈ the crowd's probability. See Karl Whelan (2025), *Agreeing to Disagree: The Economics of Betting Exchanges* — a model where participants disagree but are on average correct ([UCD WP2025_22](https://www.ucd.ie/economics/t4media/WP2025_22.pdf) · [CEPR DP20633](https://cepr.org/publications/dp20633) · [MPRA](https://mpra.ub.uni-muenchen.de/126351/)).

---

## 2) The platforms (as of 2024–2026)

Commission is charged on **net winnings** per market; rates fall with volume/rewards. **Availability is geo-restricted and legality varies by country** (see §9).

| Platform | Region | Commission (net winnings) | Public/dev API? | Notes | Link |
|---|---|---|---|---|---|
| **Betfair Exchange** | 🌍 UK/EU/AU (not most US) | Market Base Rate ~**5%** UK/EU (AU 6–10%), reduced by *Discount Rate*; **Expert Fee** replaced Premium Charge Jan 2025 | ✅ Full REST + **Stream** | Deepest **liquidity**; the reference exchange | [betfair.com](https://www.betfair.com/exchange/) · [charges](https://support.betfair.com/app/answers/detail/413-exchange-what-is-commission-and-how-is-it-calculated/) |
| **Smarkets** | 🌍 UK/EU (limited) | Flat **2%** on net winnings | ✅ HTTP API (£150 one-off access fee) | Low flat commission; clean order book | [smarkets.com](https://smarkets.com/) · [docs](https://docs.smarkets.com/) |
| **Matchbook** | 🌍 UK/EU/AU/CA | **1.5%** (0.75% on the *maker* side); "**Matchbook Zero**" 0% on select events | ✅ REST (free <1M GET/mo) | Sharp-friendly, low commission | [matchbook.com](https://www.matchbook.com/) · [api.matchbook.com](https://api.matchbook.com/) |
| **Betdaq** | 🌍 UK/IE (limited) | **2%** UK/IE/Gibraltar/Jersey (5% elsewhere); API/heavy-usage accounts may face different rates | ✅ REST/SOAP | Long-standing #2; lower liquidity | [betdaq.com](https://www.betdaq.com/) · [api.betdaq.com](https://api.betdaq.com/) |
| **ProphetX** (US) | 🇺🇸 most US states | P2P, no vig; commission on winnings | Limited | Sweepstakes-model P2P exchange 2024–25; **CFTC-approved sports prediction market** (DCM/DCO) from **Jun 2026** | [Sportico coverage](https://www.sportico.com/business/sports-betting/2025/prophetx-prediction-market-cftc-sweepstakes-1234876170/) |
| ~~Prophet Exchange~~ | 🇺🇸 NJ | — | — | First **legal US exchange** (NJ, Aug 2022); **exited May 2024**, rebranded → ProphetX | [PlayNJ](https://www.playnj.com/news/new-jersey-bids-adieu-propet-exchange-shuts-down-sports-betting/80653/) |

> Note: Betfair Exchange is **not available in most of the United States** (geo-blocked); US peer-to-peer/"exchange" style products are increasingly framed as **CFTC-regulated prediction markets** (event contracts) rather than sportsbooks. Match your platform to your jurisdiction (§9).

---

## 3) How prices form: the order book & liquidity

Each runner/outcome has a **limit order book**: a ladder of **available-to-back** prices (offers to lay, i.e. supply) and **available-to-lay** prices (offers to back, i.e. demand), each with a **£/R\$ amount available (liquidez)**. A trade executes when a **taker** accepts a resting **maker** offer — exactly like an equities/crypto exchange, but the "asset" is a contract paying 1 unit if the outcome occurs. Key mechanics:

- **Back price ≈ 1/probability.** Odds 2.0 ⇒ implied ~50%. Back and lay best prices straddle the fair price; the gap is the **bid–ask spread**.
- **Liquidity (liquidez)** concentrates on big markets (EPL Match Odds, Champions League) and **explodes in-play**; obscure markets are thin, so spreads are wide and the price is *not* a reliable consensus.
- **Betfair Starting Price (BSP)** is a reconciled starting price where all unmatched back/lay volume crosses — the closest thing to a **single crowd-consensus number** on a deep market ([BSP explained](https://caanberry.com/bsp-betfair-starting-price/)).
- **Tick sizes** vary by odds band (e.g. 0.01 below 2.0, 0.02 from 2.0–3.0), so "1 tick" of profit means different % at different prices.

---

## 4) APIs, data & open-source frameworks (verified)

The exchange ecosystem is unusually open: real REST/stream APIs, terabytes of historic order-book data, and mature Python frameworks. **All repos below were checked for existence + maintenance.**

| Tool / API | Type | Free? | API? | Maintained? | Link |
|---|---|---|---|---|---|
| **Betfair Exchange API** (Betting-NG, Accounts) | Official REST/JSON-RPC | Free (needs App Key + SSL certs + funded acct) | ✅ | ✅ Live | [developer.betfair.com](https://developer.betfair.com/) · [docs](https://docs.developer.betfair.com/) |
| **Betfair Exchange Stream API** | Low-latency market + order stream | Free | ✅ | ✅ Live | [Stream docs](https://docs.developer.betfair.com/display/1smk3cen4v3lu3yomq5qye0ni/Exchange+Stream+API) · [sample code](https://github.com/betfair/stream-api-sample-code) |
| **betfairlightweight** (`betcode-org/betfair`, aka `liampauling/betfair`) | Python wrapper for API-NG + streaming (C/Rust-accelerated) | Free (MIT) | wraps Betfair | ✅ Active — Py 3.9–3.14 | [github](https://github.com/betcode-org/betfair) · [PyPI](https://pypi.org/project/betfairlightweight/) |
| **flumine** (`betcode-org/flumine`) | Event-based **trading framework** (strategies, risk, **simulation/paper trading/backtest**) built on betfairlightweight | Free (MIT) | Betfair, Betdaq, Betconnect | ✅ **Very active** — v3.0.0 (Jun 2026), Py 3.10–3.14 | [github](https://github.com/betcode-org/flumine) · [docs](https://betcode-org.github.io/flumine/) · [PyPI](https://pypi.org/project/flumine/) |
| **Betfair Historic Data** ("The Automation Hub") | TB-scale order-book history since 2016, **JSON** stream files, 3 tiers (Basic free / Advanced / Pro) | Freemium | Historic Data API | ✅ Active | [betfair-datascientists.github.io](https://betfair-datascientists.github.io/) · [data site guide](https://betfair-datascientists.github.io/data/usingHistoricDataSite/) |
| **autoHubTutorials** (`betfair-down-under`) | Sample code for the Automation Hub tutorials (modelling → API → automation) | Free | — | ✅ Active | [github](https://github.com/betfair-down-under/autoHubTutorials) |
| **Smarkets Trading API** | Official HTTP API (order book, price history, order mgmt) | £150 one-off access fee | ✅ | ✅ Live | [docs.smarkets.com](https://docs.smarkets.com/) |
| **smk_trading_bot** (`smarkets/`) | Official example Smarkets trading bot | Free | wraps Smarkets | ⚠️ Reference/legacy | [github](https://github.com/smarkets/smk_trading_bot) |
| **Matchbook API** | Official REST (odds, exchange, orders) | Free <1M GET/mo (£100 per extra 1M) | ✅ | ✅ Live | [api.matchbook.com](https://api.matchbook.com/) · [pricing](https://developers.matchbook.com/docs/pricing) |
| **Betdaq API** | Official REST/SOAP (account/execution) | Free (API commission terms apply) | ✅ | ✅ Live | [api.betdaq.com](https://api.betdaq.com/) |

**Typical open-source stack:** `betfairlightweight` (connect + stream) → `flumine` (strategy engine, risk, order execution) → **backtest on Betfair Historic JSON** (flumine `simulated` mode) → analyse fills vs BSP to measure whether your edge survives commission. Pair with the modelling libraries on the [Open-Source Tools](./Open_Source_Tools_and_Libraries.md) page (penaltyblog/socceraction) to generate the probabilities you would trade on.

---

## 5) Trading vocabulary (educational only — *not* a strategy)

"Trading" an exchange means opening a position (back or lay) and **closing it at a different price** to lock a profit/loss across all outcomes — *before* the result is known. This is a discipline of spreads and ticks, **not** outcome prediction. Terms you will meet in the literature:

| Term (EN / PT) | Meaning |
|---|---|
| **Back / Lay** (a favor / contra) | Buy vs sell an outcome contract |
| **Scalping** (escalpelamento) | Capture **1–3 ticks** repeatedly on tiny price moves; seconds-to-a-minute holds |
| **Swing trading** | Ride a **directional move (10–40 ticks)** over minutes; fewer, larger trades |
| **Greening up / hedging** (garantir o verde / cobrir) | After a favourable move, place the opposite bet so the **profit spreads evenly across all outcomes** ([Betfair explainer](https://apps.betfair.com/learning/greening-up-applying-maths-to-hedge-your-profit/)) |
| **In-play trading** (ao vivo) | Trade the fast price swings during a match (goals, red cards, momentum) |
| **Pre-match vs in-play** | Pre-match prices drift on team-news/money flow; in-play they **jump** on events |
| **Ladder interface** | Vertical price-ladder UI (as in Bet Angel/Geeks Toy) for fast click trading |
| **Liability** (responsabilidade) | Max you can lose on a lay = (odds − 1) × stake |

Consumer trading software (client apps over the Betfair API; verify current pricing yourself):

| Software | Since | Approx. price | Notes | Link |
|---|---|---|---|---|
| **Bet Angel** | 2004 (Peter Webb) | ~£149.99/yr (tiers) | Ladders, Guardian automation, Soccer Mystic; Betfair + Betdaq | [betangel.com](https://www.betangel.com/) |
| **Geeks Toy** | 2009 | ~£60/yr | Highly customisable ladders; Betfair + Betdaq + Matchbook | [geekstoy.com](https://www.geekstoy.com/) |
| **Advanced Cymatic Trader** | — | free trial/paid | High-performance Betfair trading terminal (PIQ, ladders, Excel bot) | [cymatic.co.uk](https://www.cymatic.co.uk/) |

---

## 6) The closing line as consensus & exchange price efficiency

On a liquid market the **closing price / BSP is the market's final, most-informed consensus** — it has absorbed team news, sharp money and last-second information. Empirically, order-driven exchange prices are **semi-strong efficient**: they update swiftly and fully to news.

- **Croxson & Reade (2014), *Information and Efficiency: Goal Arrival in Soccer Betting*** (*Economic Journal* 124(575):62–91) — Betfair match-odds prices update **swiftly and fully** after a goal; little evidence of pre-goal drift → semi-strong efficiency ([CentAUR full text](https://centaur.reading.ac.uk/34884/) · [Oxford Academic](https://academic.oup.com/ej/article/124/575/62/5076978)).
- **Angelini, De Angelis & Singleton (2022), *Informational efficiency and behaviour within in-play prediction markets*** (*Int. J. Forecasting*) — 1,000+ EPL matches on Betfair; markets move toward efficiency for *expected* news but misprice *surprises* (reverse favourite–longshot bias) ([PDF](https://centaur.reading.ac.uk/98329/1/information_efficiency_angelini_de_angelis_singleton.pdf) · [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0169207021000996)).
- **Practical implication:** *beating the closing line* (CLV) is the honest yardstick of skill — and it is very hard. For CLV depth see the sibling [Bet Selection & Staking](./Bet_Selection_Staking_and_High_Odds_Analysis.md#1-the-single-number-that-matters-closing-line-value-clv) page (not repeated here).

---

## 7) Matched betting & arbitrage — an honest explanation

These are the most-searched "risk-free" ideas. They **exist**, but the edge is small, temporary, and actively hunted down.

- **Matched betting (aposta casada)** — extract the value of a **bookmaker free bet/bonus** by backing the outcome at the book and **laying the same outcome on an exchange**, so the qualifying loss is tiny and the bonus is converted to (mostly) locked cash. The edge is the **promotion**, not a market inefficiency; once the sign-up offers run out, so does the edge. Commercial services (e.g. [OddsMonkey](https://www.oddsmonkey.com/), [Outplayed](https://outplayed.com/)) automate the maths — treat them as paid tools, not guarantees.
- **Arbitrage / "arbing" (arbitragem / *surebet*)** — bet **all outcomes** across a book and an exchange (or two books) when a price discrepancy guarantees profit whatever happens. Real, but: gaps are **tiny (often <1–3%)**, close in **seconds**, need capital spread across many accounts, and one leg can be **voided or re-priced** mid-execution.
- **Why the edges close fast:** books price-in and **limit or ban winners (*gubbing*)** within days — reduced stakes, no promos, or account closure, flagged by staking patterns, market choice and withdrawal behaviour. **Exchanges** rarely limit winners (P2P), but liquidity and commission cap the realistic edge, and Betfair applies an **Expert Fee** on large sustained profit. Net: **not risk-free, not scalable, not a job.** ([gubbing overview](https://www.oddsmonkey.com/blog/matched-betting/account-restrictions/how-to-avoid-gubbing-when-matched-betting/))

---

## 8) Market microstructure: parallels to financial markets

The exchange is a genuine **limit-order-book market**, so decades of financial-microstructure theory transfer directly. This is the most fruitful *research* angle for quants entering sports data.

| Financial-market concept | Betting-exchange analogue |
|---|---|
| Limit order book (bid/ask ladder) | Available-to-back / available-to-lay ladder per runner |
| **Maker vs taker** | Poster of a limit offer vs acceptor of it — **identifiable per trade** on Betfair 1-second data |
| Bid–ask spread | Gap between best back and best lay price |
| Transaction cost / fee | **Commission on net winnings** (settlement-time cost, not per-trade) |
| Liquidity / market depth | £/R\$ available at each rung; deepest in-play on big matches |
| **Adverse selection** | Risk that a resting lay is taken by a better-informed backer (e.g. in-play insider before a goal) |
| Inventory risk | Trader's net position/liability before greening up |
| Price discovery / EMH | BSP/closing price as crowd-consensus probability |
| Contract that settles to 0/1 | A back bet is effectively a **binary option**; lay = writing it |

Why researchers love it: Betfair records the **full order book at ~1-second resolution with maker/taker labels**, a clarity "rarely available in financial-market datasets" — a natural laboratory for market microstructure. Key reads:

- **Whelan (2025)**, *Agreeing to Disagree: The Economics of Betting Exchanges* — heterogeneous beliefs, maker/taker, commission as the exchange's revenue model ([PDF](https://www.karlwhelan.com/Papers/Betfair.pdf)).
- **HBS working paper 19-057**, *Platform Competition: Betfair and the U.K. Market for Sports Betting* — network effects, liquidity as a moat, exchange vs bookmaker competition ([PDF](https://www.hbs.edu/ris/Publication%20Files/19-057_463d21a0-7f60-440f-b261-b85ff543c231.pdf)).
- Betfair-datascientists tutorial: [*Understanding Market Formation and Price Movement*](https://betfair-datascientists.github.io/tutorials/analysingAndPredictingMarketMovements/) — hands-on with real order-book data.

---

## 9) Legality varies by country — check before you touch an API

Exchanges and P2P/prediction markets sit in **different legal boxes** than sportsbooks, and rules change fast.

| Jurisdiction | Status (2024–2026) |
|---|---|
| 🇬🇧 UK / 🇮🇪 Ireland | Exchanges licensed (UKGC/Gambling Commission); Betfair/Smarkets/Betdaq/Matchbook operate |
| 🇧🇷 **Brazil** | **Lei das Bets (Lei 14.790/2023)**, regulated from **Jan 2025** by **SPA/MF**; **betting exchange (P2P) is explicitly permitted** under Art. 49 of Ordinance SPA/MF 1,231/2024. Only SPA/MF-authorised operators may serve Brazil ([ICLG 2026](https://iclg.com/practice-areas/gambling-laws-and-regulations/brazil/) · [Chambers Gaming 2025](https://practiceguides.chambers.com/practice-guides/gaming-law-2025/brazil)) |
| 🇺🇸 USA | State sports-betting is bookmaker-only in most states; **exchange-style products migrating to CFTC-regulated prediction markets** (event contracts) — e.g. ProphetX received CFTC approval (DCM/DCO) in **June 2026** after running a sweepstakes model. Betfair Exchange itself is geo-blocked in most US states |
| 🇦🇺 Australia | Betfair licensed (Northern Territory); state-based taxes raise effective commission |

**Data-scraping/API note:** many "odds APIs" reselling exchange prices are third parties, not the venue; only the official APIs above are authoritative. Respect each platform's Terms and your local law.

---

## 10) Responsible gambling (Jogo Responsável) — read this again

| Resource | Region | Link |
|---|---|---|
| **BeGambleAware** | 🌍/UK | [begambleaware.org](https://www.begambleaware.org/) |
| **GamCare** | 🌍/UK | [gamcare.org.uk](https://www.gamcare.org.uk/) |
| **Gambling Therapy** (multilingual) | 🌍 | [gamblingtherapy.org](https://www.gamblingtherapy.org/) |
| **Gamblers Anonymous** | 🌍 | [gamblersanonymous.org](https://www.gamblersanonymous.org/) |
| 🇧🇷 **Jogo Responsável (SPA/MF)** — regulator, Lei 14.790/2023 | Brazil | [gov.br/fazenda](https://www.gov.br/fazenda/pt-br/composicao/orgaos/secretaria-de-premios-e-apostas) |
| 🇧🇷 **CVV** — apoio emocional 24h, gratuito | Brazil | ligue **188** · [cvv.org.br](https://www.cvv.org.br/) |
| 🇧🇷 **Jogadores Anônimos** | Brazil | [jogadoresanonimos.com.br](https://jogadoresanonimos.com.br/) |

Exchanges and trading tools make betting **faster, cheaper and more addictive** — tighter spreads and 24/7 in-play mean more decisions per hour. Efficiency of the market means **the expected long-run return of trading is negative after commission** for the vast majority. Brazil's Lei **14.790/2023** mandates deposit limits, self-exclusion (via the SPA's centralised **Plataforma de Autoexclusão**, live Dec 2025; operators are monitored through the **SIGAP** system) and prominent responsible-gambling information. **Ludopatia (transtorno do jogo)** is a recognised health condition — treat these pages as *market education*, never as encouragement to bet.

---

**Keywords:** betting exchange, back and lay, Betfair Exchange, Smarkets, Matchbook, Betdaq, ProphetX, prediction market, betfairlightweight, flumine, Betfair Stream API, Betfair historic data, order book, limit order, maker/taker, liquidity, bid-ask spread, commission, market base rate, sports trading, scalping, swing trading, greening up, hedging, in-play trading, Betfair Starting Price (BSP), closing line value, matched betting, arbitrage/arbing, gubbing/account limits, adverse selection, market microstructure, market efficiency, responsible gambling · **Português:** bolsa de apostas, aposta a favor e contra, mercado order-driven, comissão, livro de ofertas, liquidez, spread, negociação esportiva, escalpelamento, garantir o verde, cobertura/hedge, ao vivo, preço de fechamento, aposta casada, arbitragem/surebet, limitação de conta, seleção adversa, microestrutura de mercado, eficiência de mercado, jogo responsável, ludopatia, Lei das Bets 14.790, SPA/MF, CVV 188.
