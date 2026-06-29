# China Stock Markets — Shanghai, Shenzhen, STAR & Beijing

> Authoritative reference on mainland China (onshore) equity markets — SSE, SZSE, STAR Market, ChiNext and the Beijing Stock Exchange — covering share classes, indices, trading mechanics, foreign access (Stock Connect / QFI), the ML/quant angle, and the data libraries (Tushare, AkShare, baostock) and ticker conventions used to pull A-share data. Audience: Brazil-heavy ML/quant practitioners (público brasileiro).

China runs the world's second-largest equity market by capitalization, split across **three onshore exchanges** plus offshore listings in Hong Kong. The onshore (mainland, *continental*) market trades in renminbi (CNY/RMB, *yuan*), is heavily retail-driven, and is gated by daily price-limit bands and a **T+1** settlement convention — features that materially shape any quant strategy. This page focuses on the **mainland CNY market** and how Brazilians (*brasileiros*) reach it via ETFs and B3's ETF Connect.

---

## 1. The exchanges (as bolsas)

| Exchange | Chinese | City | Founded | Boards | Regulator |
|---|---|---|---|---|---|
| Shanghai Stock Exchange (SSE) | 上海证券交易所 | Shanghai | 1990 | Main Board, **STAR Market** (科创板) | CSRC |
| Shenzhen Stock Exchange (SZSE) | 深圳证券交易所 | Shenzhen | 1990 | Main Board, **ChiNext** (创业板) | CSRC |
| Beijing Stock Exchange (BSE) | 北京证券交易所 | Beijing | **2021** (incorporated 2021-09-03; trading began **2021-11-15**) | Single board (SMEs / 专精特新) | CSRC |

- The **CSRC** (China Securities Regulatory Commission, 中国证监会) is the unified national regulator over all three exchanges, IPO registration and QFI access ([csrc.gov.cn](http://www.csrc.gov.cn/csrc_en/)).
- SSE and SZSE were both established in 1990; the **STAR Market** (Science and Technology Innovation Board) launched on SSE in July 2019 as a registration-based tech/growth board; **ChiNext** is SZSE's older growth board (2009), which moved to a registration-based regime in 2020.
- The **Beijing Stock Exchange** was announced 2 Sep 2021 and incorporated 3 Sep 2021, but **began trading on 15 Nov 2021** (81 firms in the first batch), built on the former NEEQ "Select" tier, targeting "specialized, refined, distinctive, innovative" SMEs (专精特新) ([en.wikipedia.org/wiki/Beijing_Stock_Exchange](https://en.wikipedia.org/wiki/Beijing_Stock_Exchange)).

---

## 2. Share classes (classes de ações)

Chinese issuers can have several share classes with **separate prices and currencies** for the same company — a critical gotcha for cross-listing and pairs strategies.

| Class | Where it trades | Currency | Who can buy | Notes |
|---|---|---|---|---|
| **A-shares** (A股) | SSE / SZSE / BSE | CNY (onshore RMB) | Mainland residents; foreigners via Stock Connect / QFI | The core onshore market; CSI 300, SSE Composite are A-share indices |
| **B-shares** (B股) | SSE (USD) / SZSE (HKD) | USD on SSE, HKD on SZSE | Historically foreign investors; now also domestic | Legacy, small, illiquid; SSE quotes B-shares in USD, tick **USD 0.001** |
| **H-shares** (H股) | **Hong Kong (HKEX)** | HKD | Anyone (incl. via global brokers) | Offshore listing of mainland companies; not part of the onshore CNY market |

- The **A/H premium**: the same company's A-share (onshore) and H-share (Hong Kong) often trade at materially different prices, since the two pools of capital are segmented. This is a classic relative-value (*valor relativo*) signal.
- SSE confirms a **10% daily price limit on auction trading of all A-share and B-share stocks** and quotes A-shares with a **RMB 0.01** tick, B-shares with a **USD 0.001** tick ([english.sse.com.cn/start/trading/mechanism](https://english.sse.com.cn/start/trading/mechanism/)).

---

## 3. Boards and securities-code prefixes

A security's **6-digit code prefix** tells you its board and class — essential when mapping tickers to rules.

| Board / class | Exchange | Code prefix (示例) | yfinance suffix |
|---|---|---|---|
| SSE Main Board A-shares | SSE | 600 / 601 / 603 / 605 | `.SS` |
| SSE STAR Market (科创板) | SSE | **688** | `.SS` |
| SSE B-shares | SSE | 900 | `.SS` |
| SZSE Main Board A-shares | SZSE | 000 / 001 / 002 / 003 | `.SZ` |
| SZSE ChiNext (创业板) | SZSE | **300 / 301** | `.SZ` |
| SZSE B-shares | SZSE | 200 | `.SZ` |
| Beijing Stock Exchange | BSE | **920** (new unified segment); legacy NEEQ-derived 43x / 83x / 87x / 88x | `.BJ` |

Prefix conventions per SSE/SZSE listing rules and code guides: SSE A-shares 600/601/603/605; STAR 688; SZSE main board 000/001/002/003; ChiNext 300/301 ([Shenzhen Stock Exchange — Wikipedia](https://en.wikipedia.org/wiki/Shenzhen_Stock_Exchange); [AllTick code/rules guide](https://blog.alltick.co/chinese-stock-code-formats-and-trading-rules/)). The **BSE** migrated its listings to a dedicated **920** code segment (activated April 2024, full rollout completed October 2025), replacing the earlier 6-digit NEEQ-derived codes (43x/83x/87x/88x) that had caused confusion with SSE/SZSE codes ([BSE switch to 920 codes, China.org.cn](http://www.china.org.cn/2025-10/10/content_118117100.shtml)). Yahoo Finance / `yfinance` use `.SS` for Shanghai and `.SZ` for Shenzhen (e.g. `000001.SS` = SSE Composite, `399001.SZ` = SZSE Component) ([Yahoo Finance exchanges list](https://help.yahoo.com/kb/SLN2310.html)).

---

## 4. Indices (índices)

| Index | Code | Provider | Coverage | Notes |
|---|---|---|---|---|
| **CSI 300** (沪深300) | 000300 | China Securities Index Co. (CSI) | Top 300 large-caps across SSE+SZSE | Flagship benchmark; free-float, cap-weighted; underlies index futures (CFFEX IF) |
| **SSE Composite** (上证综指) | 000001 | SSE | All A+B shares on SSE | Launched 1991-07-15; cap-weighted, base 100 |
| **SZSE Component** (深证成指) | 399001 | SZSE / CNI | 500 SZSE stocks | Main SZSE benchmark |
| **STAR 50** (科创50) | 000688 | CSI | 50 largest STAR Market names | Tech/growth proxy |
| **CSI 500** (中证500) | 000905 | CSI | Mid-cap (ranks 301–800) | Small/mid-cap factor proxy |
| SSE 50 (上证50) | 000016 | SSE/CSI | Top 50 SSE mega-caps | Subset of SSE 180 |

- **CSI 300** is compiled by **China Securities Index Co., Ltd.**, capitalization-weighted and **free-float adjusted**, using a Paasche-formula chained index (Index = adjusted mkt cap / base × 1000) ([CSI 300 Index Methodology PDF, csindex.com.cn](https://csi-web-dev.oss-cn-shanghai-finance-1-pub.aliyuncs.com/static/html/csindex/public/uploads/indices/detail/files/en/000300_Index_Methodology_en.pdf)).
- The **SSE Composite** covers all A- and B-shares on Shanghai, launched 15 Jul 1991 ([SSE Composite — Wikipedia](https://en.wikipedia.org/wiki/SSE_Composite_Index)); the **SZSE Component** tracks 500 Shenzhen stocks ([SZSE Component — Wikipedia](https://en.wikipedia.org/wiki/SZSE_Component_Index)).

---

## 5. Market mechanics (mecânica de mercado)

### 5.1 Trading hours (horário, China Standard Time, GMT+8)

| Session | Time (CST) |
|---|---|
| Opening call auction (leilão de abertura) | 09:15 – 09:25 |
| Continuous trading — morning | 09:30 – 11:30 |
| Lunch break (intervalo) | 11:30 – 13:00 |
| Continuous trading — afternoon | 13:00 – 14:57 |
| Closing call auction (leilão de fechamento) | 14:57 – 15:00 |

Hours per SSE schedule and trading-mechanism pages ([english.sse.com.cn trading schedule](https://english.sse.com.cn/start/trading/schedule/)); SZSE mirrors the same sessions. There is **no after-market continuous session**; a post-close **fixed-price (大宗/盘后) trading** window exists and, under 2025–2026 reforms, is being expanded from STAR/ChiNext to all A-shares and ETFs ([SSE 2023 rules, BigGo summary](https://finance.biggo.com/news/r84-hJ0Bh5an-7GhF5Vo)).

### 5.2 Settlement

- **T+1** for shares: stock bought on day T can only be **sold on T+1**; cash from a sale settles same-day for reinvestment but withdrawal follows T+1. Confirmed by SSE ("Stock trading implements the T+1 trading system… stocks bought on T day can only be sold on T+1 day") ([SSE trading mechanism](https://english.sse.com.cn/start/trading/mechanism/)).
- No intraday round-tripping of the *same* shares — a hard constraint on intraday mean-reversion / scalping models.

### 5.3 Daily price-limit bands (limites de oscilação diária)

| Board / instrument | Daily limit (±) | IPO first days |
|---|---|---|
| SSE & SZSE Main Board A/B-shares | **±10%** | **No price limit for first 5 trading days** post-IPO (since the 2023 registration-based reform; pre-2023 the first day was capped at +44% / −36%) |
| Risk-warning stocks (ST / *ST) | **±5%** (widening to ±10% under 2026 reform) | — |
| **STAR Market** (科创板) | **±20%** | **No price limit for first 5 trading days** |
| **ChiNext** (创业板) | **±20%** | **No price limit for first 5 trading days** |
| **Beijing Stock Exchange** | **±30%** | First day **no limit**; halt 10 min if +30% / −60% |

Sources: SSE — 10% on A/B-share auction trading, STAR ±20%, "daily price limit does not apply on the first five trading days after an IPO" ([SSE mechanism](https://english.sse.com.cn/start/trading/mechanism/)); SZSE main board ±10%, ChiNext **±20%** since 24 Aug 2020 ([effectiveness-of-price-limits study, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10289463/)); since the registration-based reform took effect on the main boards (Feb 2023) **all four boards have no price limit for the first 5 trading days** post-IPO — before that the SSE/SZSE main board first day was capped at +44% / −36% ([China IPO registration-based reform, China Briefing](https://www.china-briefing.com/news/chinas-ipo-reforms-registration-based-mechanism/)); BSE **±30%** after debut, no limit on debut with a 10-minute halt at +30%/−60% ([Beijing Stock Exchange — Wikipedia](https://en.wikipedia.org/wiki/Beijing_Stock_Exchange)). Reform note: under rules taking effect in 2026, SSE/SZSE plan to widen the **ST/*ST band from ±5% to ±10%** ([SSE 2023-rev trading rules, BigGo](https://finance.biggo.com/news/r84-hJ0Bh5an-7GhF5Vo)).

### 5.4 Order/lot conventions

- **Round lot (lote)**: buy orders in multiples of **100 shares** on the main board; **STAR Market** limit orders **≥200 and ≤100,000 shares** ([SSE mechanism](https://english.sse.com.cn/start/trading/mechanism/)).
- Max single main-board order **≤ 1,000,000 shares**; block trade (大宗交易) **≥300,000 shares or ≥ RMB 2,000,000** ([SSE mechanism](https://english.sse.com.cn/start/trading/mechanism/)).
- Sell orders may be in odd lots (the remaining shares from a partial position).

---

## 6. Foreign access (acesso de estrangeiros)

Foreigners cannot open a normal onshore brokerage account; access routes are:

| Route | What it is | Key limits |
|---|---|---|
| **Stock Connect** (沪深港通) | Mutual market access through HKEX: **Shanghai–HK** and **Shenzhen–HK** Connect. *Northbound* = HK/intl money buying A-shares; *Southbound* = mainland money buying HK stocks | **Northbound daily quota RMB 52bn** per link (A-shares + eligible ETFs share the quota); reset daily, no carry-over |
| **QFI** (合格境外投资者) | Qualified Foreign Investor — merged **QFII** (FX) + **RQFII** (offshore RMB) regimes; institutional license for broad onshore access (stocks, bonds, futures) | License-based; subject to aggregate foreign-ownership caps |
| **B3 ETF Connect** | Brazil–China ETF cross-listing on B3 (see §9) | Retail-friendly local route for *brasileiros* |

Stock Connect Northbound daily quota of **RMB 52 billion per link**, shared by A-shares and eligible ETFs, reset daily with no roll-over, per HKEX ([HKEX Stock Connect](https://www.hkex.com.hk/Mutual-Market/Connect-Hub/Stock-Connect?sc_lang=en)). Foreign holdings via Connect aggregate with QFII/RQFII for **single-stock foreign-ownership limits** ([HKEX Stock Connect FAQ](https://www.hkex.com.hk/-/media/HKEX-Market/Mutual-Market/Stock-Connect/Getting-Started/Information-Booklet-and-FAQ/FAQ/FAQ_En.pdf)). Connect **co-exists** with QFI/QDII rather than replacing it.

---

## 7. ML / quant angle (o ângulo de ML/quant)

The A-share market is structurally different from US/EU equities, which changes what models work:

- **Retail dominance**: individual investors generate the bulk of turnover, producing strong momentum/herding microstructure, sentiment-driven volume spikes, and exploitable overreaction — distinct from institution-heavy Western books ([Deep Learning multi-day turnover, arXiv 2506.06356](https://arxiv.org/html/2506.06356v1)).
- **Price-limit non-executability**: when a stock is **limit-up/limit-down (涨停/跌停)**, the printed close is *not tradable*. Naive backtests that assume you can buy the close at limit-up are silently look-ahead-biased — a recurring data-engineering trap that **dominates model architecture as a source of fake alpha** ([ML multi-factor, arXiv 2507.07107](https://arxiv.org/html/2507.07107)). Always mask limit-locked bars.
- **Factor structure**: clear **size** and **value (B/M)** premia in A-shares; Fama-French style models need a China-specific recalibration (e.g. dropping the smallest "shell-value" decile distorted by reverse-merger speculation) ([Factor Investing in China A-Share, CAIA](https://caia.org/sites/default/files/factor_investing_in_the_china_a-share_market.pdf); [Revised FF for China, ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0264999322002425)).
- **Limit-hit dynamics / cooling-off**: price limits spread volatility into subsequent days and create predictable pre-hit order-flow dynamics — a microstructure signal studied extensively for the A-share market ([cooling-off effect, arXiv 1803.09422](https://arxiv.org/pdf/1803.09422); [pre-hit dynamics, PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4395215/)).
- **Policy / event risk**: state intervention, IPO-quota changes, and regulatory campaigns are regime drivers; many quants add a "policy regime" feature or news-NLP overlay.
- **LOB / deep-learning**: limit-order-book representation learning and turnover-aware deep models are an active A-share research line ([LOB representation benchmark, arXiv 2505.02139](https://arxiv.org/pdf/2505.02139)).

---

## 8. Data & APIs (dados e APIs)

### 8.1 Open-source Python libraries

| Library | Source | Coverage | Auth | Notes |
|---|---|---|---|---|
| **Tushare** | [github.com/waditu/tushare](https://github.com/waditu/tushare) | Daily/min bars, fundamentals, index members, macro | **Token** (free tier + points system) | De-facto standard; `pro_api()` interface |
| **AkShare** | [github.com/akfamily/akshare](https://github.com/akfamily/akshare) | Very broad: equities, futures, options, bonds, HK/US, macro | **No key** (scrapes public endpoints) | Python ≥3.9 64-bit; huge function surface |
| **baostock** | [baostock.com](http://baostock.com) · PyPI `baostock` | Historical A-share bars, adjusted prices, valuation | **No key** (free, `login()/logout()`) | Good for clean adjusted OHLCV |
| **yfinance** | PyPI `yfinance` | A-shares & indices via Yahoo, suffix-based | None | `.SS` Shanghai, `.SZ` Shenzhen (limited BSE) |
| **Wind / Choice (东方财富)** | wind.com.cn / choice | Institutional-grade, full tick, fundamentals, news | **Paid/commercial** | Industry standard in Chinese funds |

### 8.2 Ticker / suffix conventions

```python
# yfinance — suffix encodes the exchange
import yfinance as yf
yf.download("600519.SS")   # Kweichow Moutai (SSE main board)
yf.download("000001.SS")   # SSE Composite Index
yf.download("300750.SZ")   # CATL (ChiNext, Shenzhen)
yf.download("399001.SZ")   # SZSE Component Index

# Tushare — bare 6-digit code + .SH / .SZ / .BJ market suffix
import tushare as ts
pro = ts.pro_api("YOUR_TOKEN")
pro.daily(ts_code="600519.SH", start_date="20240101", end_date="20241231")
pro.daily(ts_code="300750.SZ")            # ChiNext
pro.index_weight(index_code="000300.SH")  # CSI 300 constituents
```

- **Suffix gotcha**: Yahoo/`yfinance` use **`.SS`** for Shanghai; Tushare/AkShare/baostock use **`.SH`**. Shenzhen is **`.SZ`** everywhere; Beijing is **`.BJ`** (Tushare) and partially supported on Yahoo. Mixing conventions is a common bug.
- Index codes are reused: `000300` = CSI 300, `000001` = SSE Composite, `399001` = SZSE Component, `000688` = STAR 50, `000905` = CSI 500.

---

## 9. How Brazilians access (como brasileiros acessam)

| Vehicle | Listed on | Exposure | Note |
|---|---|---|---|
| **PKIN11** | B3 (ETF Connect) | **CSI 300** (onshore A-shares) | First direct A-share ETF via Brazil–China ETF Connect; debuted 26 May 2025 |
| **TECX11** | B3 (ETF Connect) | **ChiNext** growth | ETF Connect, direct onshore exposure |
| **XINA11** | B3 | MSCI China | Broad China (incl. HK/US-listed names) |
| **MCHI** | US (NASDAQ) | MSCI China large/mid | Largest US-listed China ETF; reachable via global brokers / BDR |
| **FXI** | US (NYSE Arca) | China Large-Cap (HK-listed 50) | Liquid large-cap proxy |
| **ASHR** | US | CSI 300 (A-shares, RQFII) | Direct onshore A-share access from the US |
| **BDRs** (BABA34, TENC34, BIDU34) | B3 | Single offshore names (Alibaba, Tencent, Baidu) | Receipts on US/HK-listed Chinese companies; **not** onshore A-shares |

- **ETF Connect** is a B3–China cross-listing program (CVM + Chinese-regulator approved) giving Brazilians *direct, regulated* onshore exposure; **PKIN11 (CSI 300)** and **TECX11 (ChiNext)** were the first pair, trading from **26 May 2025** ([B3 ETF Connect](https://www.b3.com.br/en_us/products-and-services/trading/equities/etf-connect-brazil-and-china.htm); [borainvestir.b3.com.br](https://borainvestir.b3.com.br/tipos-de-investimentos/renda-variavel/etfs/primeiros-etfs-apos-parceria-etf-connect-entre-brasil-e-china-ja-estao-disponiveis-na-b3/)).
- **Key distinction**: BDRs like **BABA34 / TENC34 / BIDU34** and ETFs like FXI/MCHI track **offshore** (HK/US-listed) Chinese shares — they do **not** give onshore CNY A-share exposure. Only ETF Connect (PKIN11/TECX11) and US RQFII ETFs (ASHR) reach the onshore A-share pool ([Seu Dinheiro](https://www.seudinheiro.com/2025/bolsa-dolar/china-e-brasil-trocam-acoes-etfs-ineditos-para-investir-em-acoes-chinesas-chegam-a-b3-mlim/)).

---

## 10. Quick reference / gotchas

- **T+1** kills same-day round-trips of the same shares; intraday models must respect it.
- **Limit-locked bars** (涨停/跌停) are not executable — mask them in backtests.
- **Suffix split**: Yahoo `.SS` vs Tushare/baostock `.SH` for Shanghai.
- **STAR & ChiNext = ±20%**, main board **±10%**, **Beijing ±30%**; new IPOs have **no price limit for the first 5 trading days** on the SSE/SZSE main board, STAR and ChiNext (since the 2023 registration-based reform), while **BSE** has no limit only on the debut day (with a 10-min halt at +30%/−60%).
- **A vs H vs B**: same company, different prices/currencies/pools.
- Northbound **Stock Connect** quota **RMB 52bn/day per link**; foreign single-stock caps aggregate Connect + QFI.

---

**Sources:** [SSE trading mechanism](https://english.sse.com.cn/start/trading/mechanism/) · [SSE trading schedule](https://english.sse.com.cn/start/trading/schedule/) · [SSE 2023 Trading Rules PDF](https://english.sse.com.cn/start/sserules/stocks/trading/c/10644064/files/7d100419dcca456b97cabaf2dfd3b904.pdf) · [SSE Composite (Wikipedia)](https://en.wikipedia.org/wiki/SSE_Composite_Index) · [SZSE Component (Wikipedia)](https://en.wikipedia.org/wiki/SZSE_Component_Index) · [Shenzhen Stock Exchange (Wikipedia)](https://en.wikipedia.org/wiki/Shenzhen_Stock_Exchange) · [Beijing Stock Exchange (Wikipedia)](https://en.wikipedia.org/wiki/Beijing_Stock_Exchange) · [ChiNext price-limit study (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10289463/) · [A-share trading-rule reforms (BigGo)](https://finance.biggo.com/news/r84-hJ0Bh5an-7GhF5Vo) · [China IPO registration-based reform (China Briefing)](https://www.china-briefing.com/news/chinas-ipo-reforms-registration-based-mechanism/) · [CSI 300 Methodology PDF (csindex)](https://csi-web-dev.oss-cn-shanghai-finance-1-pub.aliyuncs.com/static/html/csindex/public/uploads/indices/detail/files/en/000300_Index_Methodology_en.pdf) · [HKEX Stock Connect](https://www.hkex.com.hk/Mutual-Market/Connect-Hub/Stock-Connect?sc_lang=en) · [HKEX Stock Connect FAQ PDF](https://www.hkex.com.hk/-/media/HKEX-Market/Mutual-Market/Stock-Connect/Getting-Started/Information-Booklet-and-FAQ/FAQ/FAQ_En.pdf) · [CSRC](http://www.csrc.gov.cn/csrc_en/) · [Tushare (GitHub)](https://github.com/waditu/tushare) · [AkShare (GitHub)](https://github.com/akfamily/akshare) · [Yahoo Finance exchanges](https://help.yahoo.com/kb/SLN2310.html) · [Chinese stock code/rules (AllTick)](https://blog.alltick.co/chinese-stock-code-formats-and-trading-rules/) · [ML multi-factor A-share (arXiv 2507.07107)](https://arxiv.org/html/2507.07107) · [Deep-learning turnover A-share (arXiv 2506.06356)](https://arxiv.org/html/2506.06356v1) · [Factor Investing China A-Share (CAIA)](https://caia.org/sites/default/files/factor_investing_in_the_china_a-share_market.pdf) · [Cooling-off price limits (arXiv 1803.09422)](https://arxiv.org/pdf/1803.09422) · [LOB representation (arXiv 2505.02139)](https://arxiv.org/pdf/2505.02139) · [B3 ETF Connect](https://www.b3.com.br/en_us/products-and-services/trading/equities/etf-connect-brazil-and-china.htm) · [B3 ETF Connect (borainvestir)](https://borainvestir.b3.com.br/tipos-de-investimentos/renda-variavel/etfs/primeiros-etfs-apos-parceria-etf-connect-entre-brasil-e-china-ja-estao-disponiveis-na-b3/) · [Seu Dinheiro ETF Connect](https://www.seudinheiro.com/2025/bolsa-dolar/china-e-brasil-trocam-acoes-etfs-ineditos-para-investir-em-acoes-chinesas-chegam-a-b3-mlim/)

**Keywords:** China A-shares, Shanghai Stock Exchange, SSE, Shenzhen Stock Exchange, SZSE, STAR Market, 科创板, ChiNext, 创业板, Beijing Stock Exchange, 北交所, CSI 300, 沪深300, SSE Composite, SZSE Component, STAR 50, CSI 500, H-shares, B-shares, Stock Connect, 沪深港通, QFI, QFII, RQFII, T+1 settlement, price limit, 涨停, limit-up, CSRC, Tushare, AkShare, baostock, yfinance, ETF Connect, PKIN11, TECX11, ASHR, FXI, MCHI, BDR; (PT) mercado de ações da China, bolsa de Xangai, bolsa de Shenzhen, ações classe A, limite de oscilação diária, liquidação T+1, leilão de fechamento, acesso de estrangeiros, índices chineses, dados de mercado, como brasileiros investir na China, ETFs chineses na B3.
