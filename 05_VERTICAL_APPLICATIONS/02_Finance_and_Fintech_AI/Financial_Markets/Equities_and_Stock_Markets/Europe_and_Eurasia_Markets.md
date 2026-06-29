# European & Eurasian Stock Markets

> Authoritative, ML/quant-oriented index of equity markets across Europe and Eurasia — exchanges, flagship indices, regulators, Yahoo/Stooq ticker suffixes, and primary data sources — with notes on Brazilian access via ETFs/BDRs. Current as of 2025-2026.

This page covers markets **not indexed elsewhere** in this repo. US (NYSE/Nasdaq), Brazil (B3), India, China, Japan/HK/Korea/Taiwan, ASEAN, generic global data APIs, and Kaggle/HuggingFace/arXiv pulls are documented in their own pages — referenced here only when load-bearing.

---

## 1. Quick-reference matrix (exchange → index → regulator → ticker suffix)

| Country | Exchange (group) | Flagship index | Mid/secondary | Regulator | Yahoo suffix | Stooq suffix |
|---|---|---|---|---|---|---|
| UK | London Stock Exchange (LSEG) | FTSE 100 | FTSE 250, FTSE 350, FTSE All-Share | FCA | `.L` (also `.IL`) | `.UK` |
| France | Euronext Paris | CAC 40 | SBF 120, CAC Next 20 | AMF | `.PA` | `.FR` |
| Netherlands | Euronext Amsterdam | AEX | AMX (AEX Midcap) | AFM | `.AS` | `.NL` |
| Belgium | Euronext Brussels | BEL 20 | BEL Mid | FSMA | `.BR` | `.BE` |
| Portugal | Euronext Lisbon | PSI (PSI 20) | — | CMVM | `.LS` | `.PT` |
| Ireland | Euronext Dublin | ISEQ 20 / ISEQ Overall | — | Central Bank of Ireland | `.IR` | — |
| Italy | Euronext Milan (Borsa Italiana) | FTSE MIB | FTSE Italia Mid Cap, Star | CONSOB | `.MI` | `.IT` |
| Norway | Euronext Oslo Børs | OBX | OSEBX (benchmark) | Finanstilsynet | `.OL` | `.NO` |
| Germany | Deutsche Börse / Xetra (Frankfurt) | DAX 40 | MDAX, SDAX, TecDAX | BaFin | `.DE` (also `.F`) | `.DE` |
| Switzerland | SIX Swiss Exchange | SMI | SPI, SMIM | FINMA | `.SW` | `.CH` |
| Sweden | Nasdaq Stockholm | OMXS30 | OMXSPI, OMXSML | Finansinspektionen | `.ST` | `.SE` |
| Finland | Nasdaq Helsinki | OMXH25 | OMXHPI | Fin. Supervisory Auth. | `.HE` | `.FI` |
| Denmark | Nasdaq Copenhagen | OMXC25 | OMXCPI | Finanstilsynet (DK) | `.CO` | `.DK` |
| Iceland | Nasdaq Iceland | OMXI15 | — | Central Bank of Iceland | `.IC` | — |
| Spain | BME (Bolsa de Madrid / SIX-owned) | IBEX 35 | IBEX Medium/Small Cap | CNMV | `.MC` | `.ES` |
| Poland | Warsaw (GPW / WSE) | WIG20 | WIG30, mWIG40, sWIG80, WIG | KNF (UKNF) | `.WA` | `.PL` |
| Austria | Wiener Börse (Vienna) | ATX | ATX Prime, ATX TR | FMA | `.VI` | `.AT` |
| Greece | Athens Exchange (ATHEX) | ASE General / FTSE/ATHEX Large Cap | — | HCMC | `.AT`* | `.GR` |
| Turkey | Borsa İstanbul (BIST) | BIST 100 (XU100) | BIST 30, BIST 50 | CMB (SPK) | `.IS` | — |
| Russia | Moscow Exchange (MOEX) | MOEX Russia Index (IMOEX) | RTS Index (USD) | Bank of Russia | `.ME` | — |

\* Athens uses Yahoo `.AT` (ATHEX); note Vienna also historically mapped to `.VI`. Always verify a specific ticker — Yahoo replaces in-symbol dots with dashes (e.g. LSE `BT.A` → `BT-A.L`).

Sources: [Yahoo exchange list](https://help.yahoo.com/kb/SLN2310.html), [Stooq DB](https://stooq.com/db/).

---

## 2. United Kingdom — London Stock Exchange (LSEG)

- **Exchange:** [London Stock Exchange](https://www.londonstockexchange.com/), part of the London Stock Exchange Group (LSEG), which also owns FTSE Russell and Refinitiv data assets.
- **Indices:** [FTSE 100](https://www.londonstockexchange.com/indices/ftse-100) (100 largest blue chips), [FTSE 250](https://www.londonstockexchange.com/indices/ftse-250) (next 250), FTSE 350, FTSE All-Share. Index methodology in the [FTSE UK Index Series Ground Rules](https://www.lseg.com/content/dam/ftse-russell/en_us/documents/ground-rules/ftse-uk-index-series-ground-rules.pdf).
- **Regulator:** [Financial Conduct Authority (FCA)](https://www.fca.org.uk/). The FCA executed the **biggest listing-regime overhaul in decades** effective **29 July 2024**, collapsing the premium/standard split into a single **Equity Shares (Commercial Companies, ESCC)** category (public float requirement reduced to 10%) ([LSEG note](https://www.lseg.com/en/insights/ftse-russell/which-uk-shares-will-the-ftse-100-include-in-future)). Two follow-on FTSE UK Index Series changes matter for index pipelines: (1) **from 22 Sep 2025** the sterling-only requirement was dropped — EUR/USD-traded UK securities are now eligible, plus fast-entry threshold changes; (2) **from the June 2026 review**, free-float requirements for UK- and non-UK-incorporated companies are aligned at **≥10%** (previously 25% for non-UK), per the [FTSE Russell inclusion-criteria change](https://www.lseg.com/en/media-centre/press-releases/ftse-russell/2026/ftse-uk-index-series-inclusion-criteria-change).
- **Data quirk:** Pence-denominated quotes (GBX) — divide by 100 for GBP; common pitfall in ML pipelines.

## 3. Pan-European — Euronext

[Euronext](https://www.euronext.com/en) is the largest pan-European exchange group, operating **7 regulated markets** across France, Netherlands, Belgium, Portugal, Ireland, Italy and Norway, plus the [Optiq®](https://www.euronext.com/en/data/market-data/market-data-products) trading/data platform. Milestone: as of **22 September 2025, Euronext NV itself joined the CAC 40** ([Euronext press release](https://live.euronext.com/en/news/euronext-joins-cac-40r-milestone-decade-transformation)).

| Market | Index | Constituents | National regulator |
|---|---|---|---|
| Paris | CAC 40 | 40 | AMF (Autorité des Marchés Financiers) |
| Amsterdam | AEX | 25 | AFM |
| Brussels | BEL 20 | 20 | FSMA |
| Lisbon | PSI | ~15-18 | CMVM |
| Dublin | ISEQ 20 / ISEQ Overall | 20 / all | Central Bank of Ireland |
| Milan | FTSE MIB | 40 | CONSOB |
| Oslo | OBX (25) / OSEBX | 25 / benchmark | Finanstilsynet |

- **Index pages:** [Euronext Indices live](https://live.euronext.com/en/products/indices).
- **Data:** [Euronext Web Services](https://www.euronext.com/en/data/how-access-market-data/web-services) — REST web API for real-time/delayed/historical quotes; [Optiq MDG](https://www.euronext.com/en/data/market-data/market-data-products) bulk low-latency feed; also resold via [ICE Developer Portal](https://developer.ice.com/fixed-income-data-services/catalog/euronext) and LSEG.

## 4. Germany — Deutsche Börse / Xetra

- **Exchange:** [Frankfurt Stock Exchange / Xetra](https://www.cashmarket.deutsche-boerse.com/cash-en) (Deutsche Börse AG). Xetra is the electronic reference venue.
- **Indices:** [DAX 40](https://www.deutsche-boerse.com/dbg-en/) (expanded 30→40 in Sept 2021 post-**Wirecard** scandal, with tightened profitability/governance rules), MDAX (50 mid-caps), SDAX, TecDAX. Index families now governed by [STOXX/Qontigo](https://stoxx.com/index/mdax/).
- **Regulator:** [BaFin (Bundesanstalt für Finanzdienstleistungsaufsicht)](https://www.bafin.de/).
- **Open data (high value for ML):**
  - [**Deutsche Börse Public Dataset**](https://registry.opendata.aws/deutsche-boerse-pds/) on AWS — free **1-minute OHLCV** for every Xetra & Eurex security, S3 buckets in `eu-central-1`. ⚠️ **Marked deprecated on the AWS Open Data Registry** (provider no longer maintaining it) — historical data remains usable for backtesting, but do not expect fresh updates.
  - [**A7 (Deutsche-Boerse/a7)**](https://github.com/Deutsche-Boerse/a7) — cloud access to **order-by-order** historical data from Eurex & Xetra (co-location quality), plus the [A7 Analytics platform](https://mds.deutsche-boerse.com/mds-de/analytics/a7-analytics-platform).
  - [Deutsche Börse Developer Portal](https://developer.deutsche-boerse.com/apis) — additional market-data APIs.

## 5. Switzerland — SIX Swiss Exchange

- **Exchange:** [SIX Swiss Exchange](https://www.six-group.com/en/) (SIX Group). SIX also **owns BME (Spain)** since 2020.
- **Index:** [SMI (Swiss Market Index)](https://www.six-group.com/en/market-data/indices/switzerland/equity/smi.html) — 20 largest/most-liquid SPI names; broad benchmark is the SPI; SMIM for mid-caps. Heavily concentrated in Nestlé, Novartis, Roche.
- **Regulator:** [FINMA](https://www.finma.ch/) (ultimate authority); day-to-day market surveillance by SIX Exchange Regulation. FINMA was notably more active in 2025, closing **55 enforcement proceedings** (vs 38 in 2024) and opening ~450 investigations into unauthorised providers ([swissinfo](https://www.swissinfo.ch/eng/various/finma-closes-significantly-more-proceedings-in-2025/91289905)).

## 6. Nordics — Nasdaq Nordic + Euronext Oslo

[Nasdaq Nordic](https://www.nasdaq.com/european-market-activity/indexes) operates Stockholm, Helsinki, Copenhagen, Iceland (Oslo Børs is **Euronext-owned**, separate). Tradable benchmarks:

| Market | Tradable index | # | Broad index | Regulator |
|---|---|---|---|---|
| Stockholm | OMXS30 | 30 | OMXSPI | Finansinspektionen (SE) |
| Helsinki | OMXH25 | 25 | OMXHPI | FIN-FSA |
| Copenhagen | OMXC25 | 25 | OMXCPI | Finanstilsynet (DK) |
| Iceland | OMXI15 | 15 | OMXIPI | Central Bank of Iceland |
| Oslo (Euronext) | OBX | 25 | OSEBX | Finanstilsynet (NO) |

- Derivatives basis-trading supported on OMXS30, OMXO20, OMXC25, OMXH25, OMXSML ([Nasdaq](https://www.nasdaq.com/solutions/index-futures-on-the-nordic-markets)). Methodology PDFs: [Nordic equity indexes](https://indexes.nasdaqomx.com/docs/Methodology_NORDIC.pdf).
- Note: Yahoo marks Nordic quotes (`.ST/.HE/.CO`) as **real-time** — rare for free feeds — making them attractive for low-latency experiments.

## 7. Spain — BME (Bolsa de Madrid)

- **Exchange:** [Bolsas y Mercados Españoles (BME)](https://www.bolsasymercados.es/en/bme-exchange.html) — owns Madrid, Barcelona, Valencia, Bilbao; **SIX-owned** since 2020.
- **Index:** [IBEX 35](https://www.bolsasymercados.es/en/bme-exchange/prices-and-markets/shares/ibex-35-es0si0000005.html) — 35 most liquid Spanish stocks, cap-weighted, calculated by Sociedad de Bolsas, reviewed twice yearly. In 2025 the IBEX 35 reclaimed levels last seen in **2007**.
- **Regulator:** [CNMV (Comisión Nacional del Mercado de Valores)](https://www.cnmv.es/).

## 8. Poland — Warsaw (GPW / WSE)

- **Exchange:** [GPW — Giełda Papierów Wartościowych w Warszawie](https://www.gpw.pl/en-home), the **largest exchange in Central & Eastern Europe** by cap and volume. Operates the Main Market + **NewConnect** (growth) alternative market. ~399 listed companies; cap ~**PLN 2.5tn (≈USD 700bn)** (early 2026).
- **Indices:** WIG20 (blue chips), WIG30, mWIG40, sWIG80, and the broad WIG.
- **Regulator:** [KNF / UKNF (Polish Financial Supervision Authority)](https://www.knf.gov.pl/en/).
- **Data:** Poland is **Stooq's home market** — best free historical coverage of GPW exists on [Stooq](https://stooq.com/db/) (Stooq is a Polish service).

## 9. Russia — Moscow Exchange (MOEX) ⚠️ access-restricted

- **Exchange:** [Moscow Exchange (MOEX)](https://www.moex.com/en); regulated by the **Bank of Russia**.
- **Indices:** [MOEX Russia Index (IMOEX, RUB)](https://www.moex.com/en/index/IMOEX) and **RTS Index (USD-denominated)**.
- **⚠️ Sanctions & foreign-access restrictions (critical for any pipeline):**
  - Equities trading suspended **28 Feb – 24 Mar 2022** after the invasion of Ukraine; on resumption, **foreign investors were restricted** (repo/derivatives only) and assets held via the central depositary were effectively **frozen/trapped**.
  - **12-13 June 2024:** OFAC sanctioned MOEX (plus NCC and NSD) on 12 June; the exchange **halted USD/EUR trading on 13 June** ([RFE/RL](https://www.rferl.org/a/moscow-exchange-dollar-trades-sanctions/32991219.html), [OFAC Russia sanctions](https://ofac.treasury.gov/sanctions-programs-and-country-information/russian-harmful-foreign-activities-sanctions)).
  - The IMOEX rebounded to pre-war ~3,500 by mid-2024 driven by **domestic/repatriated capital**; the USD RTS stayed depressed.
  - **Practical note:** yfinance `.ME` data is often stale/unreliable for foreigners; most Western data vendors dropped real-time MOEX. Treat any MOEX dataset as **non-investable** for EU/US/BR persons and a compliance hazard. Use only for academic study.

## 10. Turkey — Borsa İstanbul (BIST)

- **Exchange:** [Borsa İstanbul (BIST)](https://www.borsaistanbul.com/en) — single entity since 2013 (merged ISE + gold + derivatives exchanges). Domestic cap ~**USD 385bn**.
- **Indices:** [BIST 100 (XU100)](https://www.borsaistanbul.com/en/index/xu100), BIST 30, BIST 50; a **USD-denominated XU100** variant exists for FX-adjusted analysis.
- **Regulator:** [Capital Markets Board (CMB / SPK)](https://www.borsaistanbul.com/en); BIST is a self-regulatory organization under CMB oversight (Capital Markets Law No. 6362). Note 2025 enforcement activity against coordinated manipulation.
- **ML caveat:** extreme **TRY inflation/devaluation** dominates nominal returns — work in USD or real terms, and watch frequent intraday **circuit breakers**.

## 11. Austria & Greece (smaller markets)

| | Austria | Greece |
|---|---|---|
| Exchange | [Wiener Börse (Vienna)](https://www.wienerborse.at/en/) | [Athens Exchange (ATHEX)](https://www.athexgroup.gr/) |
| Index | ATX (20 names), ATX Prime, ATX TR (total return) | ASE General Index; FTSE/ATHEX Large Cap (25) |
| Regulator | FMA (Finanzmarktaufsicht) | HCMC (Hellenic Capital Market Commission) |
| Yahoo suffix | `.VI` | `.AT` |
| Note | Vienna also operates indices for **CEE** (CECE, ROTX, PX-related) | Reclassified to **emerging** by MSCI in 2013; FTSE upgraded Greece toward developed status — verify current classification |

---

## 12. Consolidated data-source table (free + programmatic)

| Source | Coverage (Europe/Eurasia) | Access | Cost | Notes |
|---|---|---|---|---|
| [**yfinance**](https://github.com/ranaroussi/yfinance) | All listed exchanges via suffixes (§1) | Python, unofficial Yahoo | Free | Delayed (15-30 min); Nordics real-time; GBX/100 trap on `.L` |
| [**Stooq**](https://stooq.com/db/) | UK `.UK`, DE, PL (best), most EU; bulk zips | CSV download (no official API) | Free | Polish-built; deepest free GPW history; pandas-datareader `StooqDailyReader` |
| [**Deutsche Börse PDS**](https://registry.opendata.aws/deutsche-boerse-pds/) | Xetra + Eurex, **1-min OHLCV** | AWS S3 `eu-central-1` | Free | Best free intraday in Europe; **deprecated** on AWS registry (no longer maintained) — historical archive still usable |
| [**Deutsche-Boerse/a7**](https://github.com/Deutsche-Boerse/a7) | Eurex/Xetra **order-by-order** | Cloud API / SDK | Freemium | Tick-level, research-grade |
| [**Euronext Web Services**](https://www.euronext.com/en/data/how-access-market-data/web-services) | Paris/AMS/BRU/LIS/DUB/MIL/OSL | REST web API | Paid (low cost) | Official quotes + reference data |
| [**SIX**](https://www.six-group.com/en/products-services/the-swiss-stock-exchange/market-data.html) | SIX Swiss + BME (Spain) | Vendor feeds | Paid | FINMA-supervised official source |
| [**EODHD**](https://eodhd.com/list-of-stock-markets) | 70+ exchanges incl. all above; 15-20y EU history | REST API | Freemium/paid | Splits/div-adjusted OHLCV; intraday 1m/5m/1h |
| [**Tiingo**](https://www.tiingo.com/) | EOD global incl. major EU | REST API | Freemium | Strong fundamentals + news |
| [**OpenBB Platform**](https://docs.openbb.co/platform/reference/equity/price/historical) | Aggregator over yfinance/Tiingo/EODHD/FMP | Python/CLI | Free core | Single API, swappable providers |
| [**FTSE Russell / LSEG**](https://www.lseg.com/en/ftse-russell/academic-solutions) | Index constituents, GEIS Monitor List (~34k equities) | Academic program / vendor | Paid/academic | Free-float, nationality, review data |
| [**investiny**](https://github.com/alvarobartt/investiny) | Investing.com global (interim) | Python | Free | Stopgap for the broken `investpy` (Cloudflare); **itself largely unmaintained** (last release Oct 2022) — verify before relying on it |
| [STOXX/Qontigo](https://stoxx.com/) | DAX/MDAX family, STOXX 600, Euro STOXX 50 | Vendor | Paid | Pan-EU benchmark families |

> ⚠️ **investpy is effectively dead** — Investing.com's Cloudflare changes broke it ([issue #611](https://github.com/alvarobartt/investpy/issues/611)). For Investing.com-style pulls use [`investiny`](https://github.com/alvarobartt/investiny); for everything else prefer yfinance + Stooq + the official exchange APIs above.

---

## 13. ML / quant notes specific to Europe & Eurasia

- **Multi-currency, multi-timezone:** EUR (Eurozone), GBP, CHF, SEK, NOK, DKK, PLN, TRY, RUB. Always (a) align to a single trading calendar with [`pandas-market-calendars`](https://github.com/rsheftel/pandas_market_calendars) (LSE, XETR, XPAR, SIX, etc.) and (b) decide on FX-hedged vs. local-currency returns before training. Carry/FX dominates Turkish and (historically) Russian nominal series.
- **Index reconstitution as a feature/label source:** STOXX (DAX/MDAX), FTSE Russell (FTSE 100/250 quarterly reviews), and IBEX twice-yearly reviews create well-documented **add/delete events** useful for index-effect studies. FTSE UK rule changes (EUR/USD-traded eligibility + fast-entry from **Sep 2025**; UK/non-UK free-float alignment at ≥10% from the **June 2026** review) shift the FTSE 100/250 universe — retrain feature pipelines accordingly.
- **Free intraday:** the **Deutsche Börse Public Dataset (1-min)** is the standout free intraday corpus in Europe; pair with Xetra A7 tick data for microstructure work.
- **Survivorship & corporate actions:** UK pence quoting (GBX), frequent CHF/SEK splits, and Euronext cross-listings (same ISIN, multiple MICs) require careful dedup by **ISIN + primary MIC**.
- **Sanctions/compliance gate:** exclude or quarantine **MOEX (`.ME`)** data from any investable strategy; it is non-tradable for EU/US/BR persons and Western data is stale. Treat as academic-only.
- **MIC codes** (ISO 10383) to key exchanges: XLON (London), XPAR (Paris), XAMS (Amsterdam), XBRU (Brussels), XLIS (Lisbon), XDUB (Dublin), XMIL/MTAA (Milan), XOSL (Oslo), XETR (Xetra), XSWX (SIX), XSTO (Stockholm), XHEL (Helsinki), XCSE (Copenhagen), XMAD (Madrid), XWAR (Warsaw), XWBO (Vienna), XATH (Athens), XIST (Istanbul), MISX (MOEX).

---

## 14. Brazil access (acesso do investidor brasileiro)

Brazilians (investidores brasileiros) cannot trade most of these exchanges directly without an offshore broker, but get exposure via **US-listed ETFs** (accessible through offshore accounts or, for some, **BDRs** on B3) and a few BDRs of individual European names.

| Ticker | ETF | Exposure | Expense ratio | Brazil note |
|---|---|---|---|---|
| **VGK** | Vanguard FTSE Europe | Developed Europe incl. UK + Switzerland | **0.06%** | Broadest, cheapest; top-10 ≈ 20% |
| **EZU** | iShares MSCI Eurozone | Eurozone only (excludes UK/CH) | ~0.50% | Pure euro-bloc bet |
| **IEUR** | iShares Core MSCI Europe | Developed Europe (incl. UK/CH) | ~0.05-0.09% | VGK alternative |
| **EWU** | iShares MSCI United Kingdom | UK only | ~0.50% | Single-country UK |
| **EWG** | iShares MSCI Germany | Germany only | **0.49%** | Single-country DAX-ish |
| **EWL / EWN / EWP / EWD / EPOL / TUR** | iShares single-country | CH / NL / Spain / Sweden / Poland / Turkey | ~0.50-0.59% | Granular country bets |

- **BDRs (B3):** Many of these ETFs and select European blue chips are available as **BDRs** on B3 (Brazilian Depositary Receipts) — check current B3 listings, as the BDR universe changes. Individual European-company BDRs are sparser than US ones.
- **Tax/FX note:** ETF exposure carries USD/EUR FX risk on top of equity risk; Brazilian residents face IR (imposto de renda) on offshore gains and B3-BDR dividend/JCP rules — consult current Receita Federal guidance.
- Sources: [VGK Vanguard](https://investor.vanguard.com/investment-products/etfs/profile/vgk), [VGK etfdb](https://etfdb.com/etf/VGK/), iShares EWU/EWG/EZU 497K filings on [SEC EDGAR](https://www.sec.gov/cgi-bin/browse-edgar).

---

## Sources

- Yahoo Finance exchange suffixes: https://help.yahoo.com/kb/SLN2310.html
- Stooq free market data: https://stooq.com/db/ — DB index: https://stooq.com/db/h/
- LSE / FTSE: https://www.londonstockexchange.com/indices/ftse-100 · https://www.londonstockexchange.com/indices/ftse-250 · FTSE UK Ground Rules https://www.lseg.com/content/dam/ftse-russell/en_us/documents/ground-rules/ftse-uk-index-series-ground-rules.pdf · https://www.lseg.com/en/insights/ftse-russell/which-uk-shares-will-the-ftse-100-include-in-future · FTSE 2026 free-float change https://www.lseg.com/en/media-centre/press-releases/ftse-russell/2026/ftse-uk-index-series-inclusion-criteria-change · FTSE 2025 methodology change https://www.lseg.com/en/media-centre/press-releases/ftse-russell/2025/ftse-uk-index-series-methodology-changes · FCA https://www.fca.org.uk/
- Euronext: https://www.euronext.com/en · indices https://live.euronext.com/en/products/indices · CAC 40 join https://live.euronext.com/en/news/euronext-joins-cac-40r-milestone-decade-transformation · Web Services https://www.euronext.com/en/data/how-access-market-data/web-services · ICE catalog https://developer.ice.com/fixed-income-data-services/catalog/euronext
- Deutsche Börse: https://www.cashmarket.deutsche-boerse.com/cash-en · PDS (AWS) https://registry.opendata.aws/deutsche-boerse-pds/ · A7 repo https://github.com/Deutsche-Boerse/a7 · A7 platform https://mds.deutsche-boerse.com/mds-de/analytics/a7-analytics-platform · Dev portal https://developer.deutsche-boerse.com/apis · BaFin https://www.bafin.de/ · STOXX MDAX https://stoxx.com/index/mdax/
- SIX / SMI: https://www.six-group.com/en/market-data/indices/switzerland/equity/smi.html · governance https://www.six-group.com/en/company/governance/monitoring-and-regulation.html · FINMA 2025 https://www.swissinfo.ch/eng/various/finma-closes-significantly-more-proceedings-in-2025/91289905
- Nasdaq Nordic: https://www.nasdaq.com/european-market-activity/indexes · methodology https://indexes.nasdaqomx.com/docs/Methodology_NORDIC.pdf · futures https://www.nasdaq.com/solutions/index-futures-on-the-nordic-markets
- BME / IBEX 35: https://www.bolsasymercados.es/en/bme-exchange/prices-and-markets/shares/ibex-35-es0si0000005.html · CNMV https://www.cnmv.es/
- GPW / WIG20: https://www.gpw.pl/en-home · KNF https://www.knf.gov.pl/en/
- MOEX: https://www.moex.com/en · index https://www.moex.com/en/index/IMOEX · sanctions https://www.rferl.org/a/moscow-exchange-dollar-trades-sanctions/32991219.html · OFAC GL https://ofac.treasury.gov/faqs/topic/6626
- Borsa İstanbul / BIST 100: https://www.borsaistanbul.com/en/index/xu100 · indices https://www.borsaistanbul.com/en/indices
- Wiener Börse: https://www.wienerborse.at/en/ · ATHEX: https://www.athexgroup.gr/
- Data tooling: yfinance https://github.com/ranaroussi/yfinance · EODHD https://eodhd.com/list-of-stock-markets · Tiingo https://www.tiingo.com/ · OpenBB https://docs.openbb.co/platform/reference/equity/price/historical · investiny https://github.com/alvarobartt/investiny · investpy issue https://github.com/alvarobartt/investpy/issues/611 · FTSE Russell academic https://www.lseg.com/en/ftse-russell/academic-solutions
- Brazil ETFs: VGK https://investor.vanguard.com/investment-products/etfs/profile/vgk · https://etfdb.com/etf/VGK/ · SEC EDGAR https://www.sec.gov/cgi-bin/browse-edgar

**Keywords:** European stock markets, Eurasian equity markets, LSE FTSE 100, Euronext CAC 40 AEX, DAX 40 Xetra, SIX SMI, Nasdaq Nordic OMXS30, IBEX 35, WIG20 GPW, MOEX RTS sanctions, BIST 100 Borsa Istanbul, ATX ASE, yfinance suffixes, Stooq, Deutsche Börse Public Dataset, investiny, ETF VGK EZU EWU EWG, BDR — mercados de ações europeus, bolsas de valores Europa Eurásia, índices acionários, dados de mercado, sufixos de ticker, ações europeias, acesso do investidor brasileiro, ETFs e BDRs.
