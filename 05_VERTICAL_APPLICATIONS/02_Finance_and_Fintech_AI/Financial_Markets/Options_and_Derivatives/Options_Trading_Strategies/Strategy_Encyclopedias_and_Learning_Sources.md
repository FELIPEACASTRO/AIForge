# Options Strategy Encyclopedias & Learning Sources

> The "traga tudo" master source list: WHERE to find authoritative, free and paid material on every options strategy — strategy libraries, the canonical books, courses/communities, podcasts, and Brazil-specific (🇧🇷 *fontes em português*) resources — with verified working links and an honest note on what each source is good (and bad) for. Education/research only — **not investment advice**.

---

## How to use this page

This page is a **curated index of learning sources**, not a strategy tutorial. The mechanics, payoffs, Greeks, and adjustments live in the sibling pages of this folder:

- **Strategy selection & playbooks** → `Strategy_Selection_and_Playbooks.md`
- **Directional & spreads** → `Directional_and_Spread_Strategies.md`
- **Volatility / income / neutral** → `Volatility_Income_and_Neutral_Strategies.md`
- **Position management & adjustments** → `Position_Management_Adjustments_and_Risk.md`
- **Execution / strike & expiry selection** → `Execution_Strike_and_Expiry_Selection.md`
- **Hedging, tail risk & macro** → `Hedging_Tail_Risk_and_Macro_Strategies.md`
- **Systematic & ML** → `Systematic_and_ML_Options_Strategies.md`
- **Analytics & flow tools** → `../Options_Strategies_and_Analytics_Tools.md`

**Reading order for a beginner (free path):** OIC Quick Guide → Cboe Options Institute basics → Option Alpha basics course → tastylive Learn → The Options Playbook (free site) → one book (Overby or Cohen) → then Natenberg/McMillan as references. **Quant path:** Natenberg → Sinclair (*Volatility Trading* → *Positional Option Trading*) → Hull → arXiv/SSRN.

> **Quality filter — read before trusting anything below.** Exchange/clearing sources (Cboe, OIC/OCC, B3) are *unbiased and free* but generic. Broker sources (Fidelity, Schwab, tastytrade) are excellent and free but exist to get you trading. Influencer/community content is uneven — verify every "edge" claim against data. Anyone selling a course, signal service, or "guaranteed income" strategy has a conflict of interest. **No source removes assignment risk, IV crush, or tail risk** — those are properties of the instrument, not of your education.

---

## 1. Strategy encyclopedias & reference libraries (free, web)

The fastest way to look up a named strategy's setup, payoff diagram, breakeven, and "when to use it."

| Resource | Type | Free/Paid | What it's best for | Link |
|---|---|---|---|---|
| **OIC — Options Strategies Quick Guide** (OCC) | Interactive strategy screener | **Free** | Filter by *forecast* (bullish/bearish/neutral) or *objective* (income, hedge, leverage) → strategy shortlist; the most neutral library | [optionseducation.org/the-options-strategies-quick-guide](https://www.optionseducation.org/the-options-strategies-quick-guide) · [PDF](https://www.optionseducation.org/getattachment/007fe864-029a-490d-8dc1-3b58bd558f64/options-strategies-quick-guide.pdf) |
| **The Options Playbook** (Brian Overby) | Strategy library, "playbook" format | **Free** site (+ book) | 40+ "plays," each as Setup / Who Should Run It / When / The Strategy with risk & cost; very readable | [optionsplaybook.com](https://www.optionsplaybook.com/) |
| **Cboe Options Institute** | Exchange education portal | **Free** | Courses, tools, on-demand from the home of SPX/VIX; "Level Up" learning portal | [cboe.com/optionsinstitute](https://www.cboe.com/optionsinstitute/) · [Learning Portal](https://www.cboe.com/optionsinstitute/learningportal/) |
| **OIC — OptionsEducation.org** (OCC) | Industry council since 1992 | **Free** | Webinars, podcasts, white papers, FAQ, OCC Learning courses | [optionseducation.org](https://www.optionseducation.org/) · [Strategies FAQ](https://www.optionseducation.org/referencelibrary/faq/strategies) |
| **The Options & Futures Guide** | Strategy encyclopedia | **Free** (ad-supported) | Large alphabetical library with profit graphs; "Option Strategy Finder" by view/risk | [theoptionsguide.com](https://www.theoptionsguide.com/) · [Strategy Finder](https://www.theoptionsguide.com/option-trading-strategies.aspx) |
| **Option Alpha — Strategies & Handbook** | Strategy library + glossary | **Free** | 36+ strategies with diagrams; the "Handbook" is a visual options encyclopedia | [optionalpha.com/options-strategies](https://optionalpha.com/options-strategies) · [Handbook](https://optionalpha.com/handbook) |
| **tastylive — Learn Center** | Mechanics-heavy education | **Free** | Premium-selling mechanics, POP, "manage winners"; quizzes; backs up claims with studies | [tastylive Learn](https://tastylive.freshdesk.com/support/solutions/articles/48001141882-beginner-options-trading-course) · [tastytrade/learn](https://tastytrade.com/learn/) |
| **Investopedia — Options strategies hub** | Reference articles | **Free** | Clean definitions and worked examples; good first stop for a single term | [investopedia.com (options strategies)](https://www.investopedia.com/trading/options-strategies-and-tactics/) |
| **Fidelity — Options Strategy Guide** | Broker education | **Free** | Strategy-by-strategy guide + live/on-demand classes (Trading Strategy Desk) | [Fidelity strategy guide](https://www.fidelity.com/learning-center/investment-products/options/options-strategy-guide/overview) · [Education](https://www.fidelity.com/options-trading/education) |
| **Charles Schwab — Options education** | Broker education | **Free** | Beginner→advanced; thinkorswim-linked coaching; income & directional courses | [schwab.com/learn/topic/options](https://www.schwab.com/learn/topic/options) · [Strategies](https://www.schwab.com/options/options-trading-strategies) |
| **Merrill Edge — Options education** | Broker education | **Free** | Bank-of-America-aligned intro material; strategy basics | [merrilledge.com (options)](https://www.merrilledge.com/investment-products/options) |

> **Note on payoff diagrams:** libraries draw the *expiration* payoff (hockey-stick). Real positions live on the *current* P&L curve, which is smoothed by time value and IV. Always cross-check a library diagram against an interactive builder (OptionStrat / OptionsProfitCalculator — see the tools page) before risking capital.

---

## 2. The canonical books (with publisher/store links + ISBN)

Ranked roughly by depth. Editions/ISBNs verified via Amazon/Goodreads/publisher pages.

| Book | Author | Angle | Level | Link (ISBN) |
|---|---|---|---|---|
| **Options as a Strategic Investment** (5th ed.) | Lawrence G. McMillan | The 1,000-page "bible" — every strategy, margin, LEAPS, structured products | Reference / All | [Amazon 9780735204652](https://www.amazon.com/Options-as-Strategic-Investment-Fifth/dp/0735204659) · [Goodreads](https://www.goodreads.com/book/show/893157.Options_as_a_Strategic_Investment) |
| **Option Volatility & Pricing** (2nd ed.) | Sheldon Natenberg | The trader's volatility "bible" — pricing, Greeks, vol trading | Intermediate→Adv | [Amazon 9780071818773](https://www.amazon.com/Option-Volatility-Pricing-Strategies-Techniques/dp/0071818774) |
| **The Bible of Options Strategies** (2nd ed.) | Guy Cohen | 58 strategies, each as goal / outlook / vol / risk-reward / proficiency | Beginner→Inter | [Amazon 9780133964028](https://www.amazon.com/Bible-Options-Strategies-Definitive-Practical/dp/0133964027) · [Sample PDF (Pearson)](https://ptgmedia.pearsoncmg.com/images/9780133964028/samplepages/9780133964028.pdf) |
| **The Options Playbook** (40 strategies) | Brian Overby | Playbook-style, plain-English plays; companion to the free site | Beginner | [Amazon B0CV3YFBWF](https://www.amazon.com/Options-Playbook-Featuring-strategies-all-stars/dp/B0CV3YFBWF) · [optionsplaybook.com book](https://www.optionsplaybook.com/the-options-playbook---book) |
| **Trading Options Greeks** | Dan Passarelli | Delta/gamma/theta/vega as the real driver of P&L | Intermediate | [Wiley/Amazon](https://www.amazon.com/Trading-Options-Greeks-Volatility-Pricing/dp/1118133161) |
| **Volatility Trading** (2nd ed.) | Euan Sinclair | Measuring/forecasting vol, capturing the VRP, position sizing/Kelly | Advanced / Quant | [Wiley/Amazon](https://www.amazon.com/Volatility-Trading-Website-Euan-Sinclair/dp/1118347137) |
| **Positional Option Trading** | Euan Sinclair | Anomaly-by-anomaly: why an edge persists, with data | Advanced / Quant | [Amazon 9781119583516](https://www.amazon.com/Positional-Option-Trading-Wiley/dp/1119583519) |
| **Options, Futures, and Other Derivatives** | John C. Hull | The university derivatives textbook — pricing theory, risk-neutral math | Academic | [Pearson — Hull](https://www.pearson.com/en-us/subject-catalog/p/options-futures-and-other-derivatives/P200000005938) |

> **Honest book guidance.** *Cohen* or *Overby* if you want a strategy lookup; *Natenberg* if you want to actually understand vol; *McMillan* as a desk reference you'll never read cover-to-cover; *Sinclair* if you can do the math and want a real edge framework; *Hull* for the theory behind the prices. Skip the dozens of self-published "make $X/month with options" e-books — none of them survive contact with IV crush and slippage.

---

## 3. Courses, communities & podcasts

| Resource | Type | Free/Paid | Notes | Link |
|---|---|---|---|---|
| **Option Alpha — Courses** | Structured courses (basics→advanced) | **Free** | 160+ videos, no-ads, mechanical/automation bent; quizzes & tracks | [optionalpha.com/courses](https://optionalpha.com/courses) · [Education](https://optionalpha.com/education) |
| **tastylive / tastytrade** | Live network + courses | **Free** education | 10+ hrs/day live; premium-selling philosophy (Tom Sosnoff); broker is tastytrade | [tastytrade-courses](https://tastytrade.com/tastytrade-courses/) · [courses.tastytrade.com](https://courses.tastytrade.com/) |
| **Cboe Options Institute — webinars** | Live + on-demand | **Free** | Exchange-grade, SPX/VIX focus; "Level Up Learning" | [cboe.com/optionsinstitute](https://www.cboe.com/optionsinstitute/) |
| **OIC / OCC Learning** | Self-paced courses & webinars | **Free** | Unbiased, clearing-house perspective; assignment/exercise mechanics done right | [OCC Learning](https://www.optionseducation.org/theoptionseducationcenter/occ-learning) |
| **Coursera — derivatives/options** | University MOOCs | Free audit / paid cert | Academic pricing & risk (e.g., financial-engineering specializations) | [coursera.org (options)](https://www.coursera.org/search?query=options%20trading) |
| **Udemy — options trading** | Marketplace courses | **Paid** (often on sale) | Quality varies wildly — check reviews & instructor track record | [udemy.com (options)](https://www.udemy.com/topic/options-trading/) |
| **r/options** | Reddit community | **Free** | Active Q&A; strong wiki/FAQ; beware "0DTE lottery" survivorship posts | [reddit.com/r/options](https://www.reddit.com/r/options/) |
| **r/thetagang** | Reddit community (premium sellers) | **Free** | Wheel / cash-secured puts / covered calls culture; honest about tail blowups | [reddit.com/r/thetagang](https://www.reddit.com/r/thetagang/) |
| **Elite Trader** | Long-running trader forum | **Free** | Older pro/retail discussion; uneven, but deep historical threads | [elitetrader.com](https://www.elitetrader.com/) |
| **The Option Alpha Podcast** | Podcast | **Free** | Weekly concept/case-study/interview format | [optionalpha.com/podcast](https://optionalpha.com/podcast) · [Apple](https://podcasts.apple.com/us/podcast/the-option-alpha-podcast/id932492307) |
| **The tastylive network** | Daily video podcast | **Free** | Full show feed; "Option Trades Today" trade-idea segment | [Apple](https://podcasts.apple.com/us/podcast/the-tastylive-network/id496019927) · [YouTube](https://www.youtube.com/@tastyliveshow) |
| **Options Playbook Radio** (Options Insider) | Podcast | **Free** | Brian Overby walks one play per episode | [theoptionsinsider.com/shows/playbook](https://theoptionsinsider.com/shows/playbook/) |

> **Community health warning.** Reddit and forums over-represent winners and lottery payouts (survivorship bias) and under-report blown accounts. A wheel/CSP that "always works" hasn't yet met a gap-down. Treat community trade ideas as *starting points to research*, never as signals.

---

## 4. Academic & quant sources (papers, code)

For the volatility-risk-premium edge, pricing, and ML — pair with `Systematic_and_ML_Options_Strategies.md`.

| Source | Type | Notes | Link |
|---|---|---|---|
| **SSRN — derivatives** | Working papers | VRP, delta-hedged option returns, option-selling studies | [papers.ssrn.com](https://papers.ssrn.com/) |
| **arXiv q-fin (Pricing of Securities / Trading)** | Preprints | Neural pricing/hedging, deep-learning options trading | [arxiv.org/list/q-fin.PR/recent](https://arxiv.org/list/q-fin.PR/recent) |
| *Deep Learning for Options Trading: An End-To-End Approach* (Tan, Roberts, Zohren) | arXiv paper | Data-driven trading rules over option chains | [arxiv.org/abs/2407.21791](https://arxiv.org/abs/2407.21791) |
| *Neural networks for option pricing and hedging: a literature review* (Ruf, Wang) | arXiv paper | NN pricing/hedging benchmarks | [arxiv.org/abs/1911.05620](https://arxiv.org/abs/1911.05620) |
| **Quantpedia — Volatility Risk Premium Effect** | Strategy encyclopedia (academic) | Catalog of researched anomalies incl. VRP, with references | [quantpedia.com — VRP](https://quantpedia.com/strategies/volatility-risk-premium-effect) |
| **Cboe — Strategy Benchmark Indices** | Index methodology + data | BXM, PUT, CLL, CNDR, BFLY long-history return/drawdown | [cboe.com benchmark indices](https://www.cboe.com/us/indices/benchmark_indices/) |
| **GitHub** | Open-source code | `vollib`/`py_vollib` (Black-Scholes/Greeks), `mibian` | [py_vollib](https://github.com/vollib/py_vollib) · [mibian](https://github.com/yassinemaaroufi/MibianLib) |

> Papers describe *historical, often pre-cost* edges. The VRP is real but regime-dependent and pays you to hold short-vol risk that detonates in crashes (Feb 2018 "Volmageddon," Mar 2020, Aug 2024 yen-carry unwind). Backtest with realistic spreads, assignment, and margin expansion.

---

## 5. 🇧🇷 Brazil-specific sources (*fontes em português*, B3)

B3 options use Brazilian naming — *trava de alta/baixa* (vertical spreads), *borboleta* (butterfly), *condor*, *financiamento* (covered call), *compra/venda de volatilidade* (straddle/strangle) — and distinct quirks: among liquid B3 equity options, **most calls are American-style while the actively traded puts are predominantly European-style** (so early-exercise/assignment risk lives mainly on short calls), liquidity concentrates around a few underlyings (**PETR/VALE/BOVA11/index**), and expiration/settlement calendars differ from the US. B3's spec page notes options can be either style, so always confirm the individual series' style and settlement on B3.

| Resource | Type | Free/Paid | Notes | Link |
|---|---|---|---|---|
| **B3 Educação — Opções** | Exchange education | **Free** | Official courses: "Primeiros passos em Opções," "Investindo com Opções"; authoritative on B3 specs | [edu.b3.com.br — opções iniciantes](https://edu.b3.com.br/en/w/opcoes-para-iniciantes) · [Investindo com Opções](https://edu.b3.com.br/w/investindo-com-opcoes) |
| **opções.net.br** | Tools + learning | Freemium | *Simulador* de payoff/Greeks, matriz por strike/vencimento, Black-Scholes para séries B3; community studies | [opcoes.net.br](https://opcoes.net.br/) |
| **OpLab — Academy / cursos** | Platform + education | Freemium (paid tiers) | Course "A Mecânica das Opções," "o que preciso saber para começar"; real-time B3 data & strategy builder | [oplab.com.br/cursos/a-mecanica-das-opcoes](https://oplab.com.br/cursos/a-mecanica-das-opcoes/) · [oplab.com.br](https://oplab.com.br/) |
| **Portal do Trader — Introdução às Opções** | Course | **Free** | 100% free intro options course (Eduardo Becker), Portuguese | [portaldotrader.com.br — curso gratuito](https://portaldotrader.com.br/aprenda/introducao-as-opcoes-curso-gratuito) |
| **Trader Brasil — Curso de Opções e Derivativos** | Course (Black-Scholes & Gregas) | **Paid** | Hedge, volatility, Greeks; presencial/online (Prof. Flávio Lemos, CMT) | [traderbrasil.com — curso opções](https://www.traderbrasil.com/curso/curso-opcoes.php) |
| **Clube do Valor** | Blog / education | **Free** + paid | Investing education in Portuguese; options & derivatives explainers | [clubedovalor.com.br](https://clubedovalor.com.br/) |
| **Bússola do Investidor** | Fintech / education | **Free** + paid | Brazilian investing platform (since 2007); course roundups & B3 content | [bussoladoinvestidor.com.br](https://www.bussoladoinvestidor.com.br/) |
| **TradeMap — módulo de opções** | App | Free / paid tiers | Simulações, curvas de Smile/Volatilidade, Greeks, filtros (3M+ investors) | [trademap.com.br](https://trademap.com.br/) |

> **B3 reality check.** Liquidity outside the top names and front-month strikes can be thin — wide bid/ask spreads quietly eat your edge. American-style early assignment (*exercício antecipado*) is a real risk on short ITM calls before dividends/JCP. **Tax treatment is Brazil-specific and unforgiving on options:** gains are taxed at **15% (swing) / 20% (day trade)**, paid monthly via *DARF* by the last business day of the following month — and the R$20.000/month spot-equity sales exemption **does not apply to options** (every profitable month is taxable). Confirm current Receita Federal rules and consult a *contador*, not a YouTube video.

---

## 6. One-screen "where do I find X?" cheat sheet

| I need… | Go to |
|---|---|
| A diagram + breakeven for a *named* strategy | OIC Quick Guide / Options Playbook / theoptionsguide.com |
| The most *unbiased* basics | Cboe Options Institute / OIC (OCC) / B3 Educação |
| *Mechanics* of selling premium, POP, managing winners | tastylive Learn / Option Alpha |
| Deep *volatility* understanding | Natenberg → Sinclair |
| A *desk reference* for every strategy/margin | McMillan |
| A *quick lookup* book (58 strategies) | Cohen, *Bible of Options Strategies* |
| The *VRP / anomaly* research | Quantpedia, SSRN, arXiv q-fin; Sinclair |
| *Backtest benchmarks* (covered call, putwrite, condor) | Cboe BXM / PUT / CNDR / BFLY indices |
| To *visualize* my own trade | OptionStrat / OptionsProfitCalculator (see tools page) |
| 🇧🇷 *B3-specific* mechanics & data | B3 Educação, opções.net.br, OpLab |

---

## Risk & disclosure

This page is a **catalog of educational sources for research and study only — it is not investment advice, a recommendation, or a solicitation**. Options involve substantial risk, including **assignment/early exercise, IV crush after events, undefined loss on naked short options, gap and tail risk, and total loss of premium**. Past performance and backtested edges (including the volatility risk premium) do not guarantee future results and can reverse violently. Brokerage education exists partly to encourage trading; influencer/community content is uneven and conflicted. Verify every cited resource, fee, and claim independently, size positions for survival, and (in Brazil) confirm current B3 contract specs and Receita Federal tax rules with qualified professionals before trading.

**Sources:** [OIC Quick Guide](https://www.optionseducation.org/the-options-strategies-quick-guide) · [optionseducation.org](https://www.optionseducation.org/) · [Cboe Options Institute](https://www.cboe.com/optionsinstitute/) · [The Options Playbook](https://www.optionsplaybook.com/) · [theoptionsguide.com](https://www.theoptionsguide.com/) · [Option Alpha](https://optionalpha.com/) · [tastylive Learn](https://tastytrade.com/learn/) · [Investopedia options strategies](https://www.investopedia.com/trading/options-strategies-and-tactics/) · [Fidelity strategy guide](https://www.fidelity.com/learning-center/investment-products/options/options-strategy-guide/overview) · [Schwab options](https://www.schwab.com/learn/topic/options) · [Merrill Edge options](https://www.merrilledge.com/investment-products/options) · [McMillan](https://www.amazon.com/Options-as-Strategic-Investment-Fifth/dp/0735204659) · [Natenberg](https://www.amazon.com/Option-Volatility-Pricing-Strategies-Techniques/dp/0071818774) · [Cohen](https://www.amazon.com/Bible-Options-Strategies-Definitive-Practical/dp/0133964027) · [Overby/Options Playbook book](https://www.amazon.com/Options-Playbook-Featuring-strategies-all-stars/dp/B0CV3YFBWF) · [Passarelli](https://www.amazon.com/Trading-Options-Greeks-Volatility-Pricing/dp/1118133161) · [Sinclair — Volatility Trading](https://www.amazon.com/Volatility-Trading-Website-Euan-Sinclair/dp/1118347137) · [Sinclair — Positional Option Trading](https://www.amazon.com/Positional-Option-Trading-Wiley/dp/1119583519) · [Hull (Pearson)](https://www.pearson.com/en-us/subject-catalog/p/options-futures-and-other-derivatives/P200000005938) · [Option Alpha Podcast](https://optionalpha.com/podcast) · [tastylive network podcast](https://podcasts.apple.com/us/podcast/the-tastylive-network/id496019927) · [Options Playbook Radio](https://theoptionsinsider.com/shows/playbook/) · [r/options](https://www.reddit.com/r/options/) · [r/thetagang](https://www.reddit.com/r/thetagang/) · [Elite Trader](https://www.elitetrader.com/) · [Quantpedia VRP](https://quantpedia.com/strategies/volatility-risk-premium-effect) · [SSRN](https://papers.ssrn.com/) · [arXiv q-fin.PR](https://arxiv.org/list/q-fin.PR/recent) · [Deep Learning for Options Trading](https://arxiv.org/abs/2407.21791) · [Ruf & Wang — NN pricing/hedging review](https://arxiv.org/abs/1911.05620) · [Cboe benchmark indices](https://www.cboe.com/us/indices/benchmark_indices/) · [py_vollib](https://github.com/vollib/py_vollib) · [B3 Educação](https://edu.b3.com.br/en/w/opcoes-para-iniciantes) · [opções.net.br](https://opcoes.net.br/) · [OpLab Academy](https://oplab.com.br/cursos/a-mecanica-das-opcoes/) · [Portal do Trader](https://portaldotrader.com.br/aprenda/introducao-as-opcoes-curso-gratuito) · [Trader Brasil](https://www.traderbrasil.com/curso/curso-opcoes.php) · [TradeMap](https://trademap.com.br/)

**Keywords:** options strategy encyclopedia, options learning sources, The Options Playbook, OIC quick guide, Cboe Options Institute, McMillan, Natenberg, Sinclair volatility trading, tastylive, Option Alpha, volatility risk premium, covered call, iron condor; *enciclopédia de estratégias de opções, fontes de aprendizado, opções B3, trava de alta, financiamento, venda coberta, volatilidade implícita, opções.net.br, OpLab, gregas, prêmio de risco de volatilidade*
