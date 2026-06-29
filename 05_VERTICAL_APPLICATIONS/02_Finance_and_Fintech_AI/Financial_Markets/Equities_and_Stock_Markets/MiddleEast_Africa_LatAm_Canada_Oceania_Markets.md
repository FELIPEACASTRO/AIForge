# Middle East, Africa, Latin America (ex-Brazil), Canada & Oceania Markets

> Authoritative, current (2025-2026) reference for equity markets that the AIForge index does NOT cover elsewhere: GCC + Israel, frontier/emerging Africa, Spanish-speaking Latin America, Canada (TMX) and Oceania (ASX/NZX) — with exchanges, indices, regulators, data suffixes, ML/quant notes, and Brazil-access ETFs (acesso via ETFs/BDRs).

Brazil (B3), the US (NYSE/Nasdaq), India (NSE/BSE), China (SSE/SZSE), Japan/HK/Korea/Taiwan and ASEAN are documented in sibling pages of this repo and are NOT repeated here. This page fills the remaining global gaps. The first practical thing to verify before any pull: the **Yahoo Finance / `yfinance` suffix** for each venue — several differ from the obvious guess (e.g. Egypt is `.CA`, not `.EG`; both UAE venues — Abu Dhabi *and* Dubai — resolve to `.AE`, not separate codes). Note: Yahoo's own help table lists Saudi as `.SAU`, but the suffix that actually resolves on Yahoo/yfinance is `.SR` (e.g. `2222.SR` for Saudi Aramco) — always test against a live quote.

---

## 1. Quick reference — exchanges, indices, regulators, suffixes

| Region | Country | Exchange (local name) | Flagship index | Regulator | yfinance suffix |
|---|---|---|---|---|---|
| Middle East | Saudi Arabia | Saudi Exchange / **Tadawul** | TASI (Tadawul All Share) | CMA (Capital Market Authority) | `.SR` (Yahoo help lists `.SAU`) |
| Middle East | UAE — Abu Dhabi | **ADX** (Abu Dhabi Securities Exchange) | FADGI (ADX General) | SCA (Securities & Commodities Authority) | `.AE` |
| Middle East | UAE — Dubai | **DFM** (Dubai Financial Market) | DFMGI | SCA | `.AE` |
| Middle East | Qatar | **QSE** (Qatar Stock Exchange) | QE Index (QSE 20) | QFMA | `.QA` |
| Middle East | Kuwait | **Boursa Kuwait** | All Share / Premier Market | CMA Kuwait | `.KW` |
| Middle East | Israel | **TASE** (Tel-Aviv Stock Exchange) | TA-35 / TA-125 | ISA (Israel Securities Authority) | `.TA` |
| Africa | South Africa | **JSE** (Johannesburg Stock Exchange) | FTSE/JSE Top 40 (J200) | FSCA + Prudential Authority | `.JO` |
| Africa | Egypt | **EGX** (Egyptian Exchange) | EGX 30 (CASE 30) | FRA (Financial Regulatory Authority) | `.CA` |
| Africa | Nigeria | **NGX** (Nigerian Exchange) | NGX All-Share Index (ASI) | SEC Nigeria | `.LG`* |
| Africa | Kenya | **NSE** (Nairobi Securities Exchange) | NSE 20 / NSE All-Share (NASI) | CMA Kenya | `.NR`* |
| Africa | Morocco | **BVC** (Casablanca / Bourse de Casablanca) | MASI / MASI 20 | AMMC | n/a* |
| LatAm | Mexico | **BMV** (Bolsa Mexicana de Valores) + BIVA | S&P/BMV IPC (^MXX) | CNBV | `.MX` |
| LatAm | Argentina | **BYMA** (Bolsas y Mercados Argentinos) | S&P Merval | CNV | `.BA` |
| LatAm | Chile | **Bolsa de Santiago** (→ nuam) | S&P/CLX IPSA | CMF | `.SN` |
| LatAm | Colombia | **BVC** (Bolsa de Valores de Colombia → nuam) | MSCI COLCAP | SFC (Superfinanciera) | `.CL`* |
| LatAm | Peru | **BVL** (Bolsa de Valores de Lima → nuam) | S&P/BVL Peru General | SMV | `.LM`* |
| Canada | Canada | **TMX** — Toronto Stock Exchange (TSX) | S&P/TSX Composite | CIRO + CSA / provincial (OSC etc.) | `.TO` |
| Canada | Canada | **TSX Venture** (TSXV) | S&P/TSX Venture Composite | CIRO + CSA | `.V` |
| Oceania | Australia | **ASX** (Australian Securities Exchange) | S&P/ASX 200 (^AXJO) | ASIC + RBA | `.AX` |
| Oceania | New Zealand | **NZX** | S&P/NZX 50 (^NZ50) | FMA | `.NZ` |

`*` = sparse / unreliable Yahoo coverage; verify per-ticker or use a paid vendor (EODHD) or the local exchange feed. Yahoo's official suffix table: <https://help.yahoo.com/kb/finance-for-web/SLN2310.html>. Note the suffix is for *price lookup* — Yahoo fundamentals for frontier names are often blank.

---

## 2. Middle East (GCC + Israel)

The GCC ex-Saudi venues are small but index-relevant: Saudi Arabia, UAE, Qatar and Kuwait are all in **MSCI Emerging Markets**, which is why passive EM flows move them disproportionately.

| Exchange | 1-liner | Regulator | Suffix |
|---|---|---|---|
| **Saudi Tadawul** | Largest Arab market, ~US$2.7T cap; TASI fell ~13% in 2025 (weakest GCC). From **1 Feb 2026** the main market opened to *all* foreign investors — the QFI regime was abolished (confirmed by CMA, effective 1 Feb 2026). | CMA | `.SR` |
| **ADX (Abu Dhabi)** | ~90+ securities; dominated by IHC, Alpha Dhabi, FAB. FADGI is the broad index. Yahoo tickers resolve as `.AE` (e.g. `FAB.AE`). | SCA | `.AE` |
| **DFM (Dubai)** | ~65+ securities; real-estate/banking heavy (Emaar `EMAAR.AE`, Emaar Dev `EMAARDEV.AE`, DIB). ~85% of 2024 new accounts were foreign. | SCA | `.AE` |
| **QSE (Qatar)** | 54 main-market listings (Feb 2026); QE Index benchmark; QFMA issued a 2025 Code of Market Conduct. | QFMA | `.QA` |
| **Boursa Kuwait** | Premier/Main market tiers; added to MSCI EM in 2020. | CMA Kuwait | `.KW` |
| **TASE (Israel)** | Only Israeli exchange; TA-35 flagship (renamed from TA-25 in 2017), TA-125 broader. Developed-market index member. | ISA | `.TA` |

Primary sources: Saudi Exchange <https://www.saudiexchange.sa/> · CMA <https://cma.org.sa/en/> · ADX <https://www.adx.ae/> · DFM <https://www.dfm.ae/> · SCA <https://www.sca.gov.ae/> · QSE <https://www.qe.com.qa/> · QFMA <https://www.qfma.org.qa/> · Boursa Kuwait <https://www.boursakuwait.com.kw/en/> · TASE <https://www.tase.co.il/en> · ISA <https://www.isa.gov.il/>.

**Index tickers (Yahoo):** TASI `^TASI.SR`, DFM General `DFMGI.AE`, TA-35 `TA35.TA` (the ADX General / FADGI is not reliably exposed on Yahoo — pull it from ADX or a vendor). GCC trading weeks historically ran Sunday–Thursday; Saudi/UAE/Qatar/Kuwait have aligned closer to a Mon-Fri week in recent years — confirm the venue calendar before backtesting (weekend gap differs from US/EU).

---

## 3. Africa

Two derivatives milestones in 2024-2025 made Africa relevant for quant: **EGX** received its first FRA license to run securities futures (Egypt's first on-exchange derivatives), and **Casablanca (BVC)** got AMMC approval (6 May 2025) for cash-settled **MASI 20 futures**, joining JSE and Nairobi as the few African venues with on-exchange equity derivatives.

| Exchange | 1-liner | Regulator | Suffix |
|---|---|---|---|
| **JSE (Johannesburg)** | ~435 listings, ~US$1.2T cap; deepest African market; full derivatives, ETFs, bonds. FTSE/JSE Top 40 = J200. | FSCA (conduct) + Prudential Authority (SARB) | `.JO` |
| **EGX (Egypt)** | EGX 30 free-float cap-weighted; EGX 70/100 broader. Now licensed for securities futures. | FRA | `.CA` |
| **NGX (Nigeria)** | Africa's largest economy; All-Share Index (ASI) covers ~150 equities. | SEC Nigeria | `.LG`* |
| **NSE (Nairobi)** | NSE 20 (20 blue chips) + NASI (all-share); East African hub since 1954. | CMA Kenya | `.NR`* |
| **BVC (Casablanca)** | 2nd-largest African exchange (~US$116B, Aug 2025); MASI all-share + MASI 20. | AMMC | n/a* |

Primary sources: JSE <https://www.jse.co.za/> · FSCA <https://www.fsca.co.za/> · EGX <https://www.egx.com.eg/en/> · FRA <https://fra.gov.eg/> · NGX <https://ngxgroup.com/> · SEC Nigeria <https://sec.gov.ng/> · Nairobi NSE <https://www.nse.co.ke/> · CMA Kenya <https://www.cma.or.ke/> · Casablanca BVC <https://www.casablanca-bourse.com/> · AMMC <https://www.ammc.ma/en>.

**Data tip:** For Nigeria/Kenya/Morocco where Yahoo is thin, the free aggregator **african-markets.com** (<https://www.african-markets.com/>) and **afx.kwayisi.org** (NGX/GSE/USE/NSE live boards) are practical fallbacks; JSE/EGX are covered by EODHD (confirm the exact exchange codes on EODHD's live list — see §8).

---

## 4. Latin America (ex-Brazil) and the nuam integration

The big structural story: **nuam exchange** — the merger of the **Santiago (Chile)**, **Colombia (BVC)** and **Lima (Peru)** exchanges into one operator on a Nasdaq-built matching engine. Rollout is phased: **Colombia** migrated first (early 2025), **Lima/Peru** activated the unified platform in **late April 2026** (volume ~20% above the prior day on day one), with **Santiago/Chile** — the largest of the three — following. As of Q1 2026 nuam's combined market cap was **~US$532B** (up ~43% y/y) with equity-trading volumes up ~75% — making it one of the largest LatAm markets after Brazil and Mexico, targeting Mexico-scale by 2030. Indices and trading rules are being harmonized — expect ticker/index continuity breaks; re-validate symbol maps in any LatAm pipeline.

| Exchange | 1-liner | Regulator | Suffix |
|---|---|---|---|
| **BMV (Mexico)** + BIVA | S&P/BMV IPC (^MXX) closed 2025 at 64,308 (+29.9%, best since 2009); BIVA is the rival venue (FTSE-BIVA +26.6%). | CNBV | `.MX` |
| **BYMA (Argentina)** | S&P Merval benchmark (S&P DJI-administered since 2019); CNV exploring a Merval ETF for foreign listing. | CNV | `.BA` |
| **Bolsa de Santiago / nuam** | S&P/CLX IPSA = 30 most-traded Chilean names. | CMF | `.SN` |
| **BVC (Colombia) / nuam** | MSCI COLCAP = ~20 most-liquid names (benchmark since 2013). | SFC (Superintendencia Financiera) | `.CL`* |
| **BVL (Lima) / nuam** | S&P/BVL Peru General index. | SMV | `.LM`* |

MILA (Mercado Integrado Latinoamericano, 2011) was the cross-listing precursor connecting these four markets; nuam supersedes it operationally for the three Andean exchanges (Mexico stays a MILA partner but is not part of the nuam merger).

Primary sources: BMV <https://www.bmv.com.mx/> · BIVA <https://www.biva.mx/> · CNBV <https://www.gob.mx/cnbv> · BYMA <https://www.byma.com.ar/en> · CNV <https://www.argentina.gob.ar/cnv> · Bolsa de Santiago <https://www.bolsadesantiago.com/> · CMF <https://www.cmfchile.cl/> · BVC <https://www.bvc.com.co/> · BVL <https://www.bvl.com.pe/> · nuam <https://www.nuam.com/en>.

---

## 5. Canada — TMX

| Exchange | 1-liner | Regulator | Suffix |
|---|---|---|---|
| **Toronto Stock Exchange (TSX)** | Senior board; **S&P/TSX Composite** is the benchmark (financials + energy + materials heavy). | CIRO (SRO) + CSA / provincial commissions (OSC, etc.) | `.TO` |
| **TSX Venture (TSXV)** | Junior/emerging board; **S&P/TSX Venture Composite**; many mining & energy juniors. | CIRO + CSA | `.V` |
| **TSX Alpha (TSXA)** | Alternative lit/dark venue within TMX. | CIRO | (routing only) |

Canada has **no single federal securities regulator**: the **Canadian Securities Administrators (CSA)** coordinate provincial commissions, and **CIRO** (Canadian Investment Regulatory Organization, formed 2023 from IIROC+MFDA) is the national SRO overseeing trading and dealers. Many large Canadian names are interlisted on NYSE/Nasdaq (Shopify, RY, TD, ENB) — use the US line for deepest liquidity, the `.TO` line for the home book.

Primary sources: TMX/TSX <https://www.tsx.com/> · TSXV <https://www.tsx.com/en/trading/tsx-venture-exchange> · CIRO <https://www.ciro.ca/> · CSA <https://www.securities-administrators.ca/>.

---

## 6. Oceania — ASX & NZX

| Exchange | 1-liner | Regulator | Suffix |
|---|---|---|---|
| **ASX (Australia)** | **S&P/ASX 200 (^AXJO)** is the benchmark; ASX 50/300/All Ordinaries broader; strong in banks (CBA, NAB), miners (BHP, RIO), CSL. | ASIC (conduct, real-time supervision) + RBA (clearing/settlement stability) | `.AX` |
| **NZX (New Zealand)** | **S&P/NZX 50 (^NZ50)** = 50 largest free-float names; small, dividend-heavy. FMA confirmed NZX met market-operator obligations for FY ended 31 Dec 2025. | FMA | `.NZ` |

Both are developed markets in MSCI/FTSE, so they ride global passive flows. ASX is a clean, well-documented venue for quant research (good Yahoo + vendor coverage, AUD); a known structural feature is the ~10:00–16:00 AEST session with a closing single-price auction (CSPA) — model the auction print, not the last continuous trade.

Primary sources: ASX <https://www.asx.com.au/> · ASIC <https://asic.gov.au/> · NZX <https://www.nzx.com/> · FMA <https://www.fma.govt.nz/>.

---

## 7. ML / quant notes per region

| Theme | Practical guidance |
|---|---|
| **Liquidity & microstructure** | GCC ex-Saudi, Nairobi, Lima, NZX, TSXV are *thin*. Daily volume gaps and wide spreads break naïve mean-reversion/HFT signals. Use volume filters; prefer index/ETF level for backtests. |
| **Index inclusion as a factor** | MSCI EM (Saudi, UAE, Qatar, Kuwait, Egypt+frontier reviews) and FTSE upgrades drive large passive rebalance flows — a well-studied event-study/alpha signal. Track MSCI/FTSE semi-annual review dates. |
| **FX overlay** | Most local returns are dominated by currency: ARS (Argentina) and EGP (Egypt) have had sharp devaluations; ZAR, MXN, CLP are liquid EM FX. Always decompose local vs USD return; for Brazil-based investors add BRL leg. |
| **Calendar & holidays** | GCC trading weeks, Ramadan-shortened sessions, and per-country holiday maps differ from NYSE. Use `pandas_market_calendars` / `exchange_calendars` (covers ASX `XASX`, TSX `XTSE`, JSE `XJSE`, TASE `XTAE`, BMV `XMEX`, and more) before resampling. |
| **Survivorship / symbol drift** | nuam migration (2026) and frequent African delistings cause symbol breaks. Snapshot constituent lists with dates; do not assume a static universe. |
| **Sharia / screening** | Many GCC names are tracked by Sharia-compliant indices (S&P Shariah, FTSE Shariah, MSCI Islamic) — useful as an alternative screen / factor universe. |
| **Data quality** | Cross-check at least two sources for frontier prices (Yahoo vs african-markets vs exchange feed). Adjust for splits/dividends explicitly — corporate-action data is the weakest link in frontier feeds. |

Libraries that already speak these venues: **`exchange_calendars`** (<https://github.com/gerrymanoim/exchange_calendars>), **`pandas-market-calendars`** (<https://github.com/rsheftel/pandas_market_calendars>), and **`yfinance`** (<https://github.com/ranaroussi/yfinance>) for price pulls with the suffixes above.

---

## 8. Data sources

### 8.1 yfinance / Yahoo (free, suffix-based)

Use `yf.download("2222.SR")` (Saudi Aramco), `"FAB.AE"` (First Abu Dhabi Bank), `"EMAAR.AE"` (Emaar, Dubai), `"TEVA.TA"` (Teva), `"ABG.JO"` (Absa), `"WALMEX.MX"` (Walmart de México), `"SHOP.TO"` (Shopify), `"BHP.AX"` (BHP). Index lookups: `^TASI.SR`, `^MXX`, `^AXJO`, `^NZ50`, `TA35.TA`. Reliable for GCC majors, JSE, BMV, BYMA, Santiago, TSX/TSXV, ASX, NZX; **patchy** for Nigeria/Kenya/Morocco/Colombia/Peru fundamentals.

### 8.2 Local exchange feeds & national aggregators

| Venue | Free/official data |
|---|---|
| Tadawul / ADX / DFM / QSE | Market-watch pages + paid Tadawul/Refinitiv feeds (see §2 links) |
| TASE | TASE Data Hub / API <https://market.tase.co.il/en> |
| JSE | JSE market data <https://www.jse.co.za/> (paid feeds; FTSE/JSE factsheets via FTSE Russell) |
| EGX | <https://www.egx.com.eg/en/> historical & index data |
| Africa (multi) | african-markets.com, afx.kwayisi.org (NGX/NSE/GSE/USE boards) |
| Mexico | BMV/BIVA portals; S&P DJI IPC factsheet |
| nuam (CL/CO/PE) | nuam + legacy Santiago/BVC/BVL portals |
| Canada | TMX Money <https://money.tmx.com/> |
| ASX/NZX | marketindex.com.au, ASX/NZX official |

### 8.3 Commercial multi-exchange APIs (paid, best for ML pipelines)

| Vendor | Coverage relevant here | Notes |
|---|---|---|
| **EODHD** (<https://eodhd.com/>) | 70+ exchanges incl. Mexico (MX), Toronto (TO), ASX (AU), Egypt (EGX), plus GCC / Tel Aviv / JSE via their global feed | EOD + fundamentals + splits/divs; good frontier coverage. Confirm the exact exchange code per venue on the live list: <https://eodhd.com/list-of-stock-markets> |
| **Twelve Data / Marketstack / Tiingo** | Major venues (MX, TSX, ASX, JSE, TASE) | Generic global APIs — covered in this repo's data-APIs page; suffix conventions differ from Yahoo. |
| **Refinitiv / Bloomberg / FactSet** | All of the above | Institutional; authoritative corporate actions and index constituents. |
| **LSEG FTSE Russell / S&P DJI** | Index constituents & weights (FTSE/JSE Top 40, IPC, ASX 200, Merval) | Free factsheets, paid full constituents. |

---

## 9. Brazil access (acesso a partir do Brasil)

Most Brazilian investors reach these markets via **US-listed ETFs** (held through a US broker or as **BDRs** on B3 when available) rather than direct local accounts. Single-country ETFs are MSCI-based and trade on NYSE Arca/Nasdaq in USD.

| Ticker | Fund | Exposure | TER |
|---|---|---|---|
| **EWW** | iShares MSCI Mexico ETF | Mexico (BMV) | ~0.50% |
| **EZA** | iShares MSCI South Africa ETF | South Africa (JSE) | ~0.59% |
| **EWC** | iShares MSCI Canada ETF | Canada (TSX) | ~0.50% |
| **EWA** | iShares MSCI Australia ETF | Australia (ASX) | ~0.50% |
| **ENZL** | iShares MSCI New Zealand ETF | New Zealand (NZX) | ~0.50% |
| **EIS** | iShares MSCI Israel ETF | Israel (TASE) | ~0.59% |
| **KSA** | iShares MSCI Saudi Arabia ETF | Saudi (Tadawul) | ~0.74% |
| **UAE** | iShares MSCI UAE ETF | ADX + DFM | ~0.59% |
| **QAT** | iShares MSCI Qatar ETF | QSE | ~0.59% |
| **ILF** | iShares Latin America 40 ETF | Brazil + Mexico + Chile + others (LatAm large-caps) | ~0.48% |
| **ARGT** | Global X MSCI Argentina ETF | Argentina (BYMA + Argentine ADRs) | ~0.59% |

Notes & cautions: (1) **iShares Frontier and Select EM ETF (FM)** that historically held Nigeria/Kenya/Morocco names **stopped trading 6 Jan 2025 and was liquidated** (proceeds paid 9 Jan 2025) — there is now no clean single US ETF for those frontier markets; access them via ADRs/GDRs or EM-frontier blends. (2) The **WisdomTree Middle East Dividend Fund (GULF)** that previously gave broad GCC dividend exposure **was closed/liquidated around 30 Jun 2025** — it is no longer a usable route; use single-country KSA/UAE/QAT instead. (3) TERs above are approximate, last-verified figures — confirm on the issuer fact sheet before allocating. (4) Many large Canadian, Israeli and Mexican companies trade directly as **ADRs/BDRs** (e.g. RY, SHOP, TEVA, AMX/América Móvil) — often the simplest single-name route for a Brazilian portfolio.

ETF sources: iShares/BlackRock <https://www.ishares.com/> · Global X MSCI Argentina (ARGT) <https://www.globalxetfs.com/funds/argt>.

---

**Sources:** Saudi Exchange <https://www.saudiexchange.sa/>; CMA <https://cma.org.sa/en/>; ADX <https://www.adx.ae/>; DFM <https://www.dfm.ae/>; SCA <https://www.sca.gov.ae/>; QSE <https://www.qe.com.qa/>; QFMA <https://www.qfma.org.qa/>; Boursa Kuwait <https://www.boursakuwait.com.kw/en/>; TASE <https://www.tase.co.il/en>; ISA <https://www.isa.gov.il/>; JSE <https://www.jse.co.za/>; FSCA <https://www.fsca.co.za/>; FTSE/JSE Top 40 <https://research.ftserussell.com/>; EGX <https://www.egx.com.eg/en/>; FRA <https://fra.gov.eg/>; NGX <https://ngxgroup.com/>; SEC Nigeria <https://sec.gov.ng/>; Nairobi NSE <https://www.nse.co.ke/>; CMA Kenya <https://www.cma.or.ke/>; Casablanca BVC <https://www.casablanca-bourse.com/>; AMMC <https://www.ammc.ma/en>; BMV <https://www.bmv.com.mx/>; BIVA <https://www.biva.mx/>; CNBV <https://www.gob.mx/cnbv>; S&P/BMV IPC <https://www.spglobal.com/spdji/>; BYMA <https://www.byma.com.ar/en>; CNV <https://www.argentina.gob.ar/cnv>; Bolsa de Santiago <https://www.bolsadesantiago.com/>; CMF <https://www.cmfchile.cl/>; BVC Colombia <https://www.bvc.com.co/>; BVL Peru <https://www.bvl.com.pe/>; nuam <https://www.nuam.com/en>; TMX/TSX <https://www.tsx.com/>; CIRO <https://www.ciro.ca/>; CSA <https://www.securities-administrators.ca/>; ASX <https://www.asx.com.au/>; ASIC <https://asic.gov.au/>; NZX <https://www.nzx.com/>; FMA NZ <https://www.fma.govt.nz/>; Yahoo Finance suffix list <https://help.yahoo.com/kb/finance-for-web/SLN2310.html>; EODHD exchanges <https://eodhd.com/list-of-stock-markets>; iShares <https://www.ishares.com/>; Global X MSCI Argentina (ARGT) <https://www.globalxetfs.com/funds/argt>; Saudi CMA foreign-investor opening (1 Feb 2026) <https://cma.gov.sa/en/MediaCenter/NEWS/Pages/CMA_N_3974.aspx>; nuam <https://www.nuam.com/en>.

**Keywords:** emerging markets equities, frontier markets, GCC stock exchanges, Tadawul TASI, ADX DFM UAE, Qatar QSE, Boursa Kuwait, Tel Aviv TASE TA-35, Johannesburg JSE Top 40, Egyptian Exchange EGX 30, Nigerian Exchange NGX, Nairobi NSE 20, Casablanca MASI, Bolsa Mexicana IPC, BYMA Merval, IPSA Chile, COLCAP, BVL Peru, nuam exchange, MILA, TMX TSX TSXV, S&P/ASX 200, NZX 50, yfinance suffix, EODHD, country ETFs EWW EZA EWC EWA ENZL EIS KSA UAE QAT ILF ARGT; mercados emergentes, bolsas de valores, Oriente Médio, África, América Latina, Canadá, Oceania, índices acionários, reguladores de mercado, acesso via ETFs e BDRs, sufixo de dados, suíte de calendários de mercado, ações estrangeiras.
