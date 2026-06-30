# Books, Courses & Learning Resources (Markets & Quant)

> Curated, verified canon of books, courses, lectures, blogs and podcasts for learning **stocks, options, derivatives and quant/ML in finance** — free vs paid marked, with Brazil (B3 / 🇧🇷) equivalents. All URLs independently re-checked to resolve and be current (2024–2026). Complements the repo's existing pages on equity markets by region, options-market prediction, B3/US options, microstructure, backtesting and data APIs (which this page does **not** repeat).

This page is the *learning layer*: where to actually go to study the field, ranked by how practitioners and academics use these resources. It is deliberately opinionated about quality and honest about cost. Books are linked to **publisher or author pages** (not pirate mirrors); courses to the **official platform**.

---

## 1. How to use this page

| If you want to… | Start with | Section |
|---|---|---|
| Price/hedge an option from first principles | Hull → Natenberg → Gatheral | §2, §3 |
| Trade volatility for a living | Sinclair → Natenberg → Passarelli | §3 |
| Build ML trading systems in Python | Jansen → López de Prado → Chan | §4 |
| Get the math foundations | Shreve → Björk → Joshi | §2 |
| Value equities (fundamental) | Damodaran (free) → Graham | §5 |
| A free accredited degree | WorldQuant University MScFE | §6 |
| Learn in Portuguese, focused on B3 | Asimov, Quantzed, OpLab, B3 Educação | §6, §9 |
| Keep up daily/weekly | Quantocracy, Flirting with Models, q-fin SE | §7, §8 |

> **Cost honesty:** the single highest-value *free* resources here are Damodaran's full NYU courses, WorldQuant University's MScFE, MIT OCW 18.S096/18.642, the Quantopian Lecture archive, and the Caltech/Columbia Coursera audits. The highest-value *paid* are Hull, Natenberg, Sinclair and López de Prado (books), and QuantInsti EPAT / CQF (programs).

---

## 2. Derivatives, options & quant-math books (the canon)

| Book | Author | Publisher (page) | Level | Cost | Why it matters |
|---|---|---|---|---|---|
| **Options, Futures, and Other Derivatives** (11th ed., 2021) | John C. Hull | [Pearson](https://www.pearson.com/en-us/subject-catalog/p/options-futures-and-other-derivatives/P200000005938) | Intro→Inter | Paid (textbook) | The standard global reference; 11th ed. adds SOFR/post-LIBOR rates, **rough volatility**, and ML in pricing/hedging. "The Hull" is on nearly every desk and syllabus. |
| **Option Volatility & Pricing** (2nd ed., 2014) | Sheldon Natenberg | [McGraw-Hill](https://www.mheducation.com/highered/mhp/product/option-volatility-pricing-advanced-trading-strategies-techniques-2nd-edition.html) | Inter | Paid | The first book new market-makers are handed. Intuition for Greeks, vol and strategy — far more *tradeable* than academic texts. |
| **The Volatility Surface: A Practitioner's Guide** (2006) | Jim Gatheral | [Wiley](https://www.wiley.com/en-us/The+Volatility+Surface%3A+A+Practitioner%27s+Guide-p-9780471792512) | Advanced | Paid | The bridge from BSM to real surfaces: local vol, Heston, SVI, SABR, jumps. Short, dense, indispensable for vol modeling. |
| **Volatility Trading** (2nd ed., +Website, 2013) | Euan Sinclair | [Wiley](https://www.wiley.com/en-us/Volatility+Trading%2C+%2B+Website%2C+2nd+Edition-p-9781118416723) | Inter→Adv | Paid | Where statistics meets the P&L: vol forecasting, sizing, hedging, trade evaluation. Practitioner, not professor. |
| **Positional Option Trading: An Advanced Guide** (2020) | Euan Sinclair | [Wiley](https://www.wiley.com/en-us/Positional+Option+Trading%3A+An+Advanced+Guide-p-9781119583530) | Advanced | Paid | For traders who outgrew the basics: edge, directional option trades, risk as a discipline. |
| **Dynamic Hedging: Managing Vanilla and Exotic Options** (1997) | Nassim N. Taleb | [Wiley](https://www.wiley.com/en-us/Dynamic+Hedging%3A+Managing+Vanilla+and+Exotic+Options-p-9780471152804) | Advanced | Paid | The trader's-eye view of higher-order Greeks, exotics and tail risk; idiosyncratic but legendary. |
| **Paul Wilmott on Quantitative Finance** (3-vol set, 2nd ed.) | Paul Wilmott | [Wiley](https://www.wiley.com/en-gb/Paul+Wilmott+on+Quantitative+Finance%2C+3+Volume+Set%2C+2nd+Edition-p-9780470018705) | Inter→Adv | Paid | Encyclopedic PDE-centric treatment. Gentler entry: *Paul Wilmott Introduces Quantitative Finance* ([Amazon](https://www.amazon.com/Paul-Wilmott-Introduces-Quantitative-Finance/dp/0470319585)). |
| **The Concepts and Practice of Mathematical Finance** (2nd ed., 2008) | Mark S. Joshi | [Cambridge UP](https://www.cambridge.org/9780521514088) | Inter | Paid | The "how to actually implement it" companion to the math; great C++/pricing mindset. |
| **Stochastic Calculus for Finance I & II** | Steven E. Shreve | [Springer (Vol I)](https://link.springer.com/book/10.1007/978-0-387-22527-2) · [Vol II](https://link.springer.com/book/9780387401010) | Advanced | Paid | Vol I (binomial), Vol II (continuous-time). The gold-standard graduate stochastic-calculus text. |
| **Arbitrage Theory in Continuous Time** (4th ed.) | Tomas Björk | [Oxford UP](https://global.oup.com/academic/product/arbitrage-theory-in-continuous-time-9780198851615) | Advanced | Paid | Cleanest exposition of risk-neutral pricing and martingale methods. |
| **Trades, Quotes and Prices** (2018) | Bouchaud, Bonart, Donier, Gould | [Cambridge UP](https://www.cambridge.org/core/books/trades-quotes-and-prices/029A71078EE4C41C0D5D4574211AB1B5) | Advanced | Paid | Modern microstructure/market-impact reference (complements the repo's microstructure page). |
| **Trading Options Greeks** (2nd ed., 2012) | Dan Passarelli | [Amazon](https://www.amazon.com/Trading-Options-Greeks-Volatility-Pricing/dp/1118133161) | Inter | Paid | Practical Greeks: how time, vol and pricing factors drive real P&L. Good after Natenberg. |

🇧🇷 **Brazil equivalents:** B3 lists *Opções sobre Ações* and *Opções sobre Ibovespa* with American/European exercise; the conceptual canon above (Hull/Natenberg) is the standard in Brazilian MFEs and is read in English. OpLab's free e-book *Conhecendo o mercado de opções* ([PDF](https://oplab.com.br/assets/documents/eBook%20-%20Conhecendo%20o%20mercado%20de%20op%C3%A7%C3%B5es%20com%20a%20OpLab.pdf)) is a solid Portuguese on-ramp. See §9.

---

## 3. Options-trading practitioner shelf (vol & strategy)

| Resource | Author/Source | Link | Cost | Notes |
|---|---|---|---|---|
| **Option Trading: Pricing and Volatility Strategies and Techniques** (2010) | Euan Sinclair | [Wiley Online](https://onlinelibrary.wiley.com/doi/book/10.1002/9781119198673) | Paid | Sinclair's earlier, broader survey of pricing + vol techniques. |
| **The Options Playbook** | Brian Overby | [optionsplaybook.com](https://www.optionsplaybook.com/) | **Free (web)** | Strategy cookbook — 40+ structures with payoff diagrams. Great beginner reference. |
| Natenberg seminar (CME) | Sheldon Natenberg | see CME Group education | Mixed | Recorded Natenberg sessions circulate via exchanges; pair with the book. |

> **Quality caveat:** the practitioner shelf teaches *how* to trade structures; it does **not** substitute for the stochastic-calculus / surface-modeling books in §2 if you want to build pricers. Read both tracks.

---

## 4. Quant & ML-in-finance books

| Book | Author | Link | Cost | Code | Notes |
|---|---|---|---|---|---|
| **Advances in Financial Machine Learning** (2018) | Marcos López de Prado | [Wiley](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086) | Paid | partial | The most-cited modern ML-finance book: meta-labeling, fractional differentiation, purged/embargoed CV, sample weights. |
| **Machine Learning for Asset Managers** (2020) | Marcos López de Prado | [Cambridge UP](https://www.cambridge.org/core/books/machine-learning-for-asset-managers/6D9211305EA2E425D33A9F38D0AE3545) | Paid | yes | First in the *Cambridge Elements in Quantitative Finance*; concise, denoising, clustering, the "false strategy" theorem. |
| **Machine Learning for Algorithmic Trading** (2nd ed., 2020) | Stefan Jansen | [Packt](https://www.packtpub.com/en-us/product/machine-learning-for-algorithmic-trading-9781839217715) | Paid | **notebooks (free)** | Best end-to-end Python ML-trading course-in-a-book. Code (MIT, ~19k★, actively maintained): [stefan-jansen/machine-learning-for-trading](https://github.com/stefan-jansen/machine-learning-for-trading) (3rd-ed work in progress; 2nd-ed on `second-edition` branch). |
| **Quantitative Trading** (2nd ed., 2021) | Ernest P. Chan | [Wiley](https://www.wiley.com/en-us/Quantitative+Trading%3A+How+to+Build+Your+Own+Algorithmic+Trading+Business%2C+2nd+Edition-p-9781119800064) | Paid | snippets | How to bootstrap a one-person algo business; pragmatic. |
| **Algorithmic Trading: Winning Strategies and Their Rationale** | Ernest P. Chan | [Amazon](https://www.amazon.com/Algorithmic-Trading-Winning-Strategies-Rationale/dp/1118460146) | Paid | yes | Mean-reversion & momentum with statistical tests; readable. |
| **Active Portfolio Management** (2nd ed.) | Grinold & Kahn | [Amazon](https://www.amazon.com/Active-Portfolio-Management-Quantitative-Controlling/dp/0070248826) | Paid | – | The "Fundamental Law of Active Management" (IR = IC·√breadth). 2019 sequel: *Advances in Active Portfolio Management*. |
| **Inside the Black Box** (2nd ed., 2013) | Rishi K. Narang | [Wiley](https://www.wiley.com/en-us/Inside+the+Black+Box%3A+A+Simple+Guide+to%C2%A0Quantitative+and+High-Frequency+Trading%2C+2nd+Edition-p-9781118662717) | Paid | – | Plain-English map of how quant/HFT shops actually work. Best non-technical overview. |
| **Python for Finance** (2nd ed., 2018) | Yves Hilpisch | [O'Reilly](https://www.oreilly.com/library/view/python-for-finance/9781492024323/) | Paid | yes | Standard Python-finance reference. Same author: *Reinforcement Learning for Finance* (O'Reilly, 2024), code: [yhilpisch/rl4f](https://github.com/yhilpisch/rl4f). |
| **Systematic Trading** / **Advanced Futures Trading Strategies** | Robert Carver | [author's books page](https://qoppac.blogspot.com/p/books.html) | Paid | [pysystemtrade](https://github.com/pst-group/pysystemtrade) | Rigorous rules-based systematic framework + open-source engine (active; repo moved from `robcarver17/` to the `pst-group` org in Jan 2026 — old URL still redirects). See blog in §7. |

> **Limitation worth knowing:** López de Prado's books assume you already know ML and Python; start with Jansen if you don't. Backtest-overfitting warnings in *Advances* are the most important chapters most readers skip.

---

## 5. Equities / valuation / investing classics

| Resource | Author | Link | Cost | Notes |
|---|---|---|---|---|
| **The Intelligent Investor** (rev. ed., Zweig commentary) | Benjamin Graham | [Amazon](https://www.amazon.com/Intelligent-Investor-Definitive-Investing-Essentials/dp/0060555661) | Paid | The value-investing foundation; Buffett's "by far the best book on investing." |
| **Damodaran Online** (valuation everything) | Aswath Damodaran | [pages.stern.nyu.edu/~adamodar](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/home.htm) | **Free** | Lecture notes, spreadsheets, datasets, full course webcasts. The single best free valuation resource on the web (updated through 2025–2026). |
| **What Works on Wall Street** (4th ed.) | James O'Shaughnessy | [Amazon](https://www.amazon.com/What-Works-Wall-Street-Performing/dp/0071625763) | Paid | Large-scale empirical study of equity factors/strategies; data-driven investing classic. |

🇧🇷 Damodaran's material is widely used in Brazilian CFA prep and MBA finance; nothing in Portuguese matches its depth. For local valuation, pair with B3 Educação (§9).

---

## 6. Structured courses, programs & degrees

| Program | Provider | Link | Cost | Format | Notes |
|---|---|---|---|---|---|
| **MSc in Financial Engineering** | WorldQuant University | [wqu.edu/mscfe](https://www.wqu.edu/mscfe) | **Free, accredited** | Online, 2 yr, 9 courses + capstone | DEAC-accredited, tuition-free. Best free credential in the field. |
| **EPAT — Executive Programme in Algorithmic Trading** | QuantInsti | [quantinsti.com/epat](https://www.quantinsti.com/epat) | Paid | 6-mo instructor-led | Industry-built, Python, placement support; covers equities, options, futures. |
| **Quantra** (self-paced) | QuantInsti | [quantra.quantinsti.com](https://quantra.quantinsti.com/) | Mixed (free + paid) | Self-paced, Python | Many courses across stocks/options/derivatives; good à-la-carte entry before EPAT. |
| **CQF — Certificate in Quantitative Finance** | CQF Institute (Fitch Learning) | [cqf.com](https://www.cqf.com/) | Paid (see [program fees](https://www.cqf.com/about-cqf/program-fees/fees)) | 6-mo part-time, online | Practitioner quant qualification; ML & derivatives modules. Free knowledge arm: [CQF Institute](https://www.cqfinstitute.org/). |
| **Pricing Options with Mathematical Models** | Caltech (Coursera) | [coursera.org/learn/pricing-options-with-mathematical-models](https://www.coursera.org/learn/pricing-options-with-mathematical-models) | **Free to audit** | MOOC | Binomial → Brownian → BSM → stochastic vol. Best free options-pricing course. |
| **Machine Learning for Trading (CS 7646)** | Georgia Tech (OMSCS) | [omscs.gatech.edu/cs-7646-machine-learning-trading](https://omscs.gatech.edu/cs-7646-machine-learning-trading) | Paid (degree) / free notes | Graduate | Materials & syllabus public at [lucylabs.gatech.edu/ml4t](https://lucylabs.gatech.edu/ml4t/). |
| **Financial Engineering & Risk Management** | Columbia (Coursera) | [coursera.org/specializations/financialengineering](https://www.coursera.org/specializations/financialengineering) | **Free to audit** | Specialization (5 courses) | Derivative pricing, term-structure/credit, optimization, advanced pricing, computational methods. |
| **Advanced Valuation with Damodaran** | NYU Stern Exec Ed | [execed.stern.nyu.edu](https://execed.stern.nyu.edu/products/advanced-valuation-with-aswath-damodaran) | Paid | Exec ed | Paid version of the free online class for certification. |
| **18.S096 / 18.642 — Math w/ Applications in Finance** | MIT OCW | [18.642 (Fall 2024)](https://ocw.mit.edu/courses/18-642-topics-in-mathematics-with-applications-in-finance-fall-2024/) · [18.S096 (2013)](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/) | **Free** | OCW | Linear algebra → stochastic processes → VaR → Itô → Black-Scholes. 2024 refresh available. |

> Note: the older MIT 15.401 *Finance Theory* (Andrew Lo) is no longer hosted as a standalone OCW course; 18.S096/18.642 is the current canonical free MIT path for quant-finance math.

🇧🇷 **Brazil:** B3 Educação + QuantInsti launched a **free** course *Introduction to Momentum Strategies Using Python* (Oct 2024) — see [QuantInsti announcement](https://blog.quantinsti.com/collaboration-between-b3-quantinsti-empower-traders-brazil/). Local Portuguese programs in §9.

---

## 7. Free lectures, blogs & open courseware

| Resource | What it is | Link | Cost |
|---|---|---|---|
| **Quantopian Lecture Series (archive)** | 56 notebook lessons + videos on stats/factors/algos (platform closed; lectures preserved) | [GitHub gist archive](https://gist.github.com/ih2502mk/50d8f7feb614c8676383431b056f4291) | **Free** |
| **Hudson & Thames** | ML-in-finance research group implementing López de Prado; maintains MlFinLab, ArbitrageLab, PortfolioLab | [hudsonthames.org](https://hudsonthames.org/) · [github.com/hudson-and-thames](https://github.com/hudson-and-thames) | Mixed |
| **QuantStart** | Tutorials on backtesting, time series, derivatives; QSTrader engine | [quantstart.com](https://www.quantstart.com/) | **Free** (+ paid) |
| **Robert Carver — "Investment Idiocy"** | Deep, honest systematic-trading writing | [qoppac.blogspot.com](https://qoppac.blogspot.com/) | **Free** |
| **Ernie Chan — Quantitative Trading** | Long-running algo/ML-trading blog; now PredictNow.ai | [epchan.blogspot.com](http://epchan.blogspot.com/) · [predictnow.ai](https://predictnow.ai/) | **Free** (+ paid) |
| **Wilmott** | Quant community: large forum + magazine since 2001 | [wilmott.com](https://www.wilmott.com/) | **Free** (mag paid) |
| **Quantitative Finance Stack Exchange** | The Q&A site for derivatives/quant math | [quant.stackexchange.com](https://quant.stackexchange.com/) | **Free** |

---

## 8. Podcasts & newsletters (keep current)

| Show | Host / Source | Focus | Link |
|---|---|---|---|
| **Flirting with Models** | Corey Hoffstein (Newfound Research) | Systematic strategy designers; momentum, managed futures, vol | [Apple](https://podcasts.apple.com/us/podcast/flirting-with-models/id1402620531) · [thinknewfound.com](https://www.thinknewfound.com/) |
| **Chat With Traders** | founded by Aaron Fifield; now hosted by Tessa Dao | Interviews across discretionary + systematic styles | [chatwithtraders.com](https://chatwithtraders.com/) |
| **Quantocracy** | Curated quant-blog mashup | Daily/weekly aggregation of the best quant posts | [quantocracy.com](https://quantocracy.com/) |

> Quantocracy is the highest-leverage single subscription: it surfaces the best of the entire quant blogosphere (100+ blogs), including most blogs in §7.

---

## 9. 🇧🇷 Brazil-focused (Portuguese) — B3, options & quant

| Resource | What it is | Link | Cost |
|---|---|---|---|
| **B3 Educação** | Official B3 financial-education hub: opções, derivativos, day trading, certificações | [edu.b3.com.br/w/opcoes](https://edu.b3.com.br/w/opcoes) | **Free / low** |
| **OpLab** | Options-analytics platform for B3: vol heatmaps, 30+ strategies, opportunity explorer + free e-book | [oplab.com.br](https://oplab.com.br/) | Mixed (freemium) |
| **Asimov Academy — Trilha Trading Quantitativo** | Python + Data Science quant track (backtesting, optimization) in Portuguese | [asimov.academy/trading-quantitativo](https://asimov.academy/trading-quantitativo/) | Paid |
| **QuantBrasil** | Validated strategies, backtests, market tools for B3 ("não opere no escuro") | [quantbrasil.com.br](https://quantbrasil.com.br/) | Mixed |
| **Quantzed** | Quant trading education incl. *Operando como Institucional* (with an options module) | [quantzed.com.br](https://quantzed.com.br/) | Paid |
| **Trader Brasil — Curso de Opções e Derivativos** | Black-Scholes & Gregas in Portuguese | [traderbrasil.com](https://www.traderbrasil.com/) | Paid |

> **Honest note (🇧🇷):** the Brazilian ecosystem is strong on *platform + applied courses* (OpLab, Asimov, Quantzed, QuantBrasil) but thin on rigorous Portuguese-language derivatives theory — for pricing/vol modeling, Brazilian quants still read Hull, Natenberg, Gatheral and Shreve in English. Use the local resources for B3 microstructure, tributação and execution; use §2/§4 for the math.

---

## 10. Suggested learning path

1. **Foundations (free):** Damodaran Online (valuation) + Caltech "Pricing Options" (Coursera, audit) + MIT 18.642 (OCW).
2. **Options depth:** Hull → Natenberg → Gatheral; trade-craft via Sinclair + Passarelli.
3. **Quant/ML build-out:** Jansen (code along) → López de Prado (avoid overfitting) → Chan/Carver (systematize).
4. **Credential (optional):** WorldQuant MScFE (free) or EPAT/CQF (paid).
5. **Stay current:** Quantocracy daily + Flirting with Models + q-fin Stack Exchange.
6. **🇧🇷 Localize:** B3 Educação + OpLab for B3 options mechanics; Asimov/Quantzed for Portuguese applied quant.

---

**Sources:** [Pearson/Hull](https://www.pearson.com/en-us/subject-catalog/p/options-futures-and-other-derivatives/P200000005938) · [Natenberg/McGraw-Hill](https://www.mheducation.com/highered/mhp/product/option-volatility-pricing-advanced-trading-strategies-techniques-2nd-edition.html) · [Gatheral/Wiley](https://www.wiley.com/en-us/The+Volatility+Surface%3A+A+Practitioner%27s+Guide-p-9780471792512) · [Sinclair/Wiley](https://www.wiley.com/en-us/Positional+Option+Trading%3A+An+Advanced+Guide-p-9781119583530) · [Joshi/Cambridge](https://www.cambridge.org/9780521514088) · [Shreve/Springer](https://link.springer.com/book/10.1007/978-0-387-22527-2) · [Trades, Quotes and Prices/Cambridge](https://www.cambridge.org/core/books/trades-quotes-and-prices/029A71078EE4C41C0D5D4574211AB1B5) · [López de Prado/Cambridge](https://www.cambridge.org/core/books/machine-learning-for-asset-managers/6D9211305EA2E425D33A9F38D0AE3545) · [Jansen/GitHub](https://github.com/stefan-jansen/machine-learning-for-trading) · [Chan/Wiley](https://www.wiley.com/en-us/Quantitative+Trading%3A+How+to+Build+Your+Own+Algorithmic+Trading+Business%2C+2nd+Edition-p-9781119800064) · [Carver/pysystemtrade](https://github.com/pst-group/pysystemtrade) · [Damodaran/NYU](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/home.htm) · [WorldQuant University](https://www.wqu.edu/mscfe) · [QuantInsti EPAT](https://www.quantinsti.com/epat) · [CQF](https://www.cqf.com/) · [Caltech/Coursera](https://www.coursera.org/learn/pricing-options-with-mathematical-models) · [Georgia Tech ML4T](https://omscs.gatech.edu/cs-7646-machine-learning-trading) · [Columbia/Coursera](https://www.coursera.org/specializations/financialengineering) · [MIT OCW 18.642](https://ocw.mit.edu/courses/18-642-topics-in-mathematics-with-applications-in-finance-fall-2024/) · [Quantopian archive](https://gist.github.com/ih2502mk/50d8f7feb614c8676383431b056f4291) · [Hudson & Thames](https://hudsonthames.org/) · [QuantStart](https://www.quantstart.com/) · [Carver/qoppac](https://qoppac.blogspot.com/) · [Quantocracy](https://quantocracy.com/) · [Flirting with Models](https://podcasts.apple.com/us/podcast/flirting-with-models/id1402620531) · [Chat With Traders](https://chatwithtraders.com/) · [quant.stackexchange](https://quant.stackexchange.com/) · [B3 Educação](https://edu.b3.com.br/w/opcoes) · [OpLab](https://oplab.com.br/) · [Asimov Academy](https://asimov.academy/trading-quantitativo/) · [QuantBrasil](https://quantbrasil.com.br/) · [Quantzed](https://quantzed.com.br/)

**Keywords:** options books, derivatives textbooks, Hull Options Futures, Natenberg volatility, Gatheral volatility surface, Euan Sinclair, quant finance courses, machine learning for trading, López de Prado, Stefan Jansen, WorldQuant University free MScFE, QuantInsti EPAT, CQF, Damodaran valuation, MIT OpenCourseWare finance, Coursera options pricing, Quantocracy, Flirting with Models, Chat With Traders, quant Stack Exchange — livros de opções, derivativos, finanças quantitativas, curso de opções B3, trading quantitativo, precificação de opções, Black-Scholes, gregas (Greeks), volatilidade, valuation, aprendizado de máquina em finanças, OpLab, Asimov Academy, QuantBrasil, Quantzed, B3 Educação
