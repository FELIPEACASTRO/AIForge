# ASEAN & Southeast-Asia Markets — SGX, IDX, SET, Bursa, PSE, HOSE

> Reference for the six core Southeast-Asian equity venues — Singapore (SGX), Indonesia (IDX), Thailand (SET), Malaysia (Bursa), the Philippines (PSE) and Vietnam (HOSE/HNX) — covering exchanges, flagship indices, regulators, market mechanics (price limits, settlement, lot sizes), the ML/quant angle, and the data sources / Python tickers a quant in Brazil actually needs.

ASEAN equities are a heterogeneous block: one developed market (Singapore), four large emerging markets (Indonesia, Thailand, Malaysia, Philippines), and one frontier-to-emerging market under reclassification (Vietnam). They share T+2-style settlement, lot-based trading, and daily price-limit "circuit breakers" on individual names — but differ sharply in foreign-ownership rules, currency convertibility, and data accessibility. For a Brazilian (Brazil-based) investor the practical access route is regional ETFs and, occasionally, BDRs (Brazilian Depositary Receipts) on B3, since direct local accounts are hard to open.

---

## 1. Markets at a glance

| Country | Exchange | Flagship index | Regulator | Currency | yfinance suffix | Settlement |
|---|---|---|---|---|---|---|
| Singapore | SGX (Singapore Exchange) | Straits Times Index (STI) | MAS (Monetary Authority of Singapore) | SGD | `.SI` | T+2 |
| Indonesia | IDX / Bursa Efek Indonesia | IDX Composite (IHSG) | OJK (Otoritas Jasa Keuangan) | IDR | `.JK` | T+2 |
| Thailand | SET (Stock Exchange of Thailand) | SET Index, SET50/SET100 | SEC Thailand | THB | `.BK` | T+2 |
| Malaysia | Bursa Malaysia | FTSE Bursa Malaysia KLCI | Securities Commission Malaysia (SC) | MYR | `.KL` | T+2 |
| Philippines | PSE (Philippine Stock Exchange) | PSEi (PSE Composite) | SEC Philippines | PHP | `.PS` | T+2 |
| Vietnam | HOSE (Ho Chi Minh) & HNX (Hanoi) | VN-Index, VN30 | SSC (State Securities Commission) | VND | (use `vnstock`) | T+2 |

> Note on suffixes: yfinance/Yahoo uses `.SI`, `.JK`, `.BK`, `.KL`, `.PS`. Vietnam tickers are *not* reliably on Yahoo; use the `vnstock` library instead (see §9).

---

## 2. Singapore — SGX

- **Exchange:** Singapore Exchange (SGX), a vertically integrated venue (cash equities, derivatives, FX, clearing, depository/CDP). Regulator: **MAS** (Monetary Authority of Singapore), which is both central bank and securities regulator.
- **Hours (cash equities):** continuous session 09:00–12:00 and 13:00–17:00 (SGT), with pre-open/non-cancel auction phases around the open and a closing auction. <https://www.sgx.com/>
- **Settlement:** **T+2** for SGX cash equities; shares are credited to / debited from the investor's **CDP** (Central Depository) account. (CapitalMarkets.SG SGX guide, <https://capitalmarkets.sg/guides/sgx-trading-explained.php>)
- **Flagship index — Straits Times Index (STI):** the **top 30** companies in the FTSE ST All-Share Index universe by **free-float-adjusted market capitalisation**, calculated and managed by **FTSE Russell** under a joint venture (SPH Media, SGX and FTSE). A liquidity screen (median daily turnover threshold) applies, and the index is **reviewed quarterly (March, June, September and December)**. <https://www.sgx.com/indices/products/sti> · <https://www.lseg.com/en/ftse-russell/indices/sgx-st> · ground rules: <https://www.lseg.com/content/dam/ftse-russell/en_us/documents/ground-rules/straits-times-index-ground-rules.pdf>
- **SGX as Asia's derivatives hub:** SGX runs the broadest pan-Asian equity-index futures suite, including **FTSE China A50 Index Futures** (offshore China A-share exposure) <https://www.sgx.com/derivatives/products/chinaa50> and **STI futures** <https://www.sgx.com/derivatives/products/sgxsti>.
- **SGX Nifty → GIFT Nifty history (important):** SGX's flagship **SGX Nifty** (offshore Indian index future) was **migrated to NSE International Exchange (NSE IX) in GIFT City, Gujarat, India** and rebranded **GIFT Nifty** on **3 July 2023**, under an SGX–NSE connectivity agreement signed in July 2022. SGX clearing members continue to participate via the GIFT Connect link. <https://en.wikipedia.org/wiki/GIFT_Nifty> · <https://www.sgx.com/derivatives/products/gift-connect>
- **Why it matters for ML/quant:** SGX is the cleanest, most liquid, English-language, USD/SGD-friendly access point to Asian beta and is the de-facto offshore hedging venue for China A50 and (historically) India.

---

## 3. Indonesia — IDX (Bursa Efek Indonesia)

- **Exchange:** Indonesia Stock Exchange (IDX / *Bursa Efek Indonesia*), formed by the **30 November 2007 merger** of the Jakarta (JSX) and Surabaya (BES) exchanges; roots trace to a Dutch-colonial Batavia bourse founded 1912. Regulator: **OJK** (*Otoritas Jasa Keuangan*). <https://en.wikipedia.org/wiki/Indonesia_Stock_Exchange>
- **Currency / lot:** IDR; **round lot = 100 shares**.
- **Hours (Jakarta, WIB):** morning 09:00–12:00 (Mon–Thu) / 09:00–11:30 (Fri), afternoon 13:30–15:50 (with pre-open and pre-close auctions). <https://en.wikipedia.org/wiki/Indonesia_Stock_Exchange>
- **Settlement:** **T+2** (regular market), netted and book-entered by **KPEI** (clearing) into **KSEI** (central depository) accounts. <https://www.idx.co.id/en/news/tplus2-settlement/>
- **Auto Rejection (price limits):** IDX's **Auto Rejection** mechanism caps single-day moves. Post-pandemic the **lower limit (Auto Rejection Bawah) was set to 15%** across the Main, Development and New Economy boards plus ETFs and DIRE (REITs); upper limits and tighter bands apply by price tier. (AEI / IDX) <https://aei.or.id/en/press-release/understanding-auto-rejection-in-the-indonesia-stock-exchange-mechanisms-and-post-pandemic-adjustments>
- **Index-level circuit breakers:** if the **IDX Composite (IHSG)** falls **>8%** intraday → 30-min halt; further fall **>15%** → another 30-min halt; **>20%** → trading suspended. (IDX) <https://aei.or.id/en/press-release/understanding-auto-rejection-in-the-indonesia-stock-exchange-mechanisms-and-post-pandemic-adjustments>
- **Flagship index — IDX Composite (IHSG):** all-share, market-cap-weighted benchmark (Indonesian: *Indeks Harga Saham Gabungan*). Sub-indices include LQ45 (45 liquid large caps) and IDX30.

---

## 4. Thailand — SET (Stock Exchange of Thailand)

- **Exchange:** Stock Exchange of Thailand (SET); growth board is **mai**. Regulator: **SEC Thailand** (the SET is the operator, the SEC the statutory regulator).
- **Hours (Bangkok, ICT):** pre-open ~09:30 then random open between **09:55–10:00**; morning session to **12:30**; pre-open ~13:30, afternoon session **~14:00–16:30**; pre-close to a random close **16:35–16:40** then off-hour trading to 17:00. Matching is **AOM** (Automated Order Matching) during sessions, auction during pre-open/pre-close. <https://www.set.or.th/en/market/information/trading-procedure/trading-hours>
- **Settlement:** **T+2** (shifted from T+3 effective **2 March 2018**).
- **Board lot:** **100 units**; reduced to **50 units** for securities priced **≥ THB 500** for 6 consecutive months. Tick sizes are tiered (THB 0.01 below 2 baht up to THB 2.00 above 400 baht). <https://www.set.or.th/en/trading-units-tick-sizes-price-limits>
- **Price limits (ceiling/floor):** daily moves capped at **±30%** of the prior close (**±60%** for foreign-national-held securities); on the **first trading day** ceiling/floor may not exceed **3× the IPO price**. The normal security-specific **Dynamic Price Band is ±10%** of the last executed price. During stress SET has twice imposed temporary measures in 2025, each narrowing ceiling/floor to **±15%** and the dynamic price band to **±5%** (with a temporary short-selling ban): first over **8–11 April 2025** (US "reciprocal tariff" turmoil), and again from **23 June 2025** (Middle East conflict), which was lifted and **normal ±30%/±10% rules restored effective 25 June 2025**. <https://www.set.or.th/en/trading-units-tick-sizes-price-limits> · <https://mondovisione.com/media-and-resources/news/set-to-implement-temporary-measures-on-ceiling-and-floor-limits-dynamic-price-ban-202547/> · <https://mondovisione.com/media-and-resources/news/set-to-restore-normal-trading-rules-on-ceiling-and-floor-limits-and-dynamic-pri-2025624/>
- **Flagship indices:** **SET Index** (all common stocks, market-cap weighted), **SET50** (top 50) and **SET100** (top 100), reviewed **semi-annually**. SET migrated SET50/SET100 (and SETHD, SETCLMV, SETWB, SETTHSI) to **free-float-adjusted** market-cap weighting (the `...FF` variants), rebalanced in June and December. <https://www.set.or.th/en/market/index/set50/profile>

---

## 5. Malaysia — Bursa Malaysia

- **Exchange:** Bursa Malaysia (securities + derivatives, the latter via Bursa Malaysia Derivatives Berhad). Regulator: **Securities Commission Malaysia (SC)**; rulebook at <https://www.bursamalaysia.com/regulation/securities/rules_of_bursa_malaysia_securities>.
- **Currency / lot:** MYR; **board lot = 100 units**.
- **Settlement:** **T+2** (implemented **29 April 2019**). Sellers deliver securities by ~11:30 MYT on T+2 via Bursa Depository; payment by ~16:00 MYT on T+2. Failed board-lot trades face a buying-in window (~14:00–17:00). <https://www.bursamalaysia.com/trade/post_trade/securities-clearing-and-settlement/overview>
- **Flagship index — FTSE Bursa Malaysia KLCI:** the **largest 30** Main-Market companies by full market cap meeting FTSE Bursa Malaysia ground rules; **value-weighted, free-float-adjusted**, with a liquidity screen, **recalculated/disseminated in real time every 15 seconds**. <https://www.bursamalaysia.com/trade/our_products_services/indices/ftse_bursa_malaysia_indices/ftse_bursa_malaysia_klci>
- **Notable:** Malaysia is a global hub for **Islamic (Shariah-compliant) equities and sukuk**; FTSE Bursa Malaysia also publishes Shariah indices — relevant for screening-based strategies.

---

## 6. Philippines — PSE

- **Exchange:** The Philippine Stock Exchange, Inc. (PSE). Regulator: **SEC Philippines**. Clearing/settlement via **SCCP** (Securities Clearing Corporation of the Philippines, a wholly-owned PSE subsidiary). <https://www.pse.com.ph/>
- **Currency / lot:** PHP; trading uses a **board-lot table** where the minimum lot size varies by price tier (e.g. multiples of 100 shares around the PHP 10 price level). <https://www.pse.com.ph/investing-at-pse/>
- **Hours (Manila, PHT):** pre-open from ~09:15, continuous trading ~09:30–12:00, midday recess, afternoon ~13:30–15:00 with a run-off/closing phase. (TradingHours) <https://www.tradinghours.com/markets/pse>
- **Settlement:** **T+2** via SCCP (migrated to T+2 from T+3, live **24 August 2023**). <https://www.pse.com.ph/sccp-marks-successful-migration-to-the-t2-settlement-cycle/>
- **Price limits — "Static Threshold":** a **±50% daily trading band** from the last close (or last adjusted close); on hitting the ceiling/floor the stock is **frozen** unless a company/regulator disclosure justifies the move. A separate **Dynamic Threshold** halts a stock on large intraday swings. <https://documents.pse.com.ph/wp-content/uploads/sites/15/2021/04/Revised-Trading-Rules.pdf>
- **Flagship index — PSEi:** the **PSE Composite Index of 30** companies selected on free float, liquidity and capitalisation criteria; free-float-adjusted, market-cap-weighted.

---

## 7. Vietnam — HOSE & HNX

- **Exchanges:** **HOSE** (Ho Chi Minh City Stock Exchange — large caps, VN-Index) and **HNX** (Hanoi Stock Exchange — mid/small caps, plus the **UPCoM** unlisted-public market). Both sit under **VNX** (Vietnam Exchange holding). Regulator: **SSC** (State Securities Commission). <https://en.wikipedia.org/wiki/Ho_Chi_Minh_City_Stock_Exchange>
- **Currency:** VND (partially convertible; foreign-ownership limits / "foreign room" caps apply per stock).
- **Price band:** **±7%** for HOSE stocks and fund certificates; **±20% on the first trading day** of a new listing. (HNX uses a wider ±10% band.) Bonds are exempt. <https://en.wikipedia.org/wiki/Ho_Chi_Minh_City_Stock_Exchange>
- **Settlement:** **T+2** (Vietnam moved to T+2 from the older T+3-style cycle). HOSE's long-delayed **KRX (Korea Exchange) trading platform went live on 5 May 2025**, lifting capacity and laying groundwork for features (e.g. shorter settlement, short-selling) that are pre-requisites for an MSCI/FTSE EM upgrade. <https://www.regulationasia.com/vietnams-krx-trading-system-goes-live>
- **Flagship indices:** **VN-Index** (all HOSE common stocks; base 100 on 28 July 2000) and **VN30** (top-30 by cap + liquidity within VNAllshare). <https://en.wikipedia.org/wiki/Ho_Chi_Minh_City_Stock_Exchange>
- **Reclassification angle:** Vietnam is FTSE Russell **Frontier** with a watch-list path to **Secondary Emerging**; a successful upgrade is a major flows catalyst — a recurring event-driven theme for quants.

---

## 8. ASEAN-wide initiatives, FX & how Brazilians access these markets

- **Regional integration:** the **ASEAN Exchanges** collaboration links SGX, IDX, SET, Bursa, PSE and HOSE/HNX for harmonised disclosure and cross-promotion <https://www.aseanexchanges.org/>. The historical **ASEAN Trading Link** (SGX–Bursa–SET cross-border routing) has been wound down, but cross-listing and depositary-receipt programmes persist.
- **FX considerations:** returns are **multi-currency** (SGD, IDR, THB, MYR, PHP, VND). IDR/VND carry the most FX volatility and capital-control friction; SGD is fully convertible. Always model **local-currency vs USD-hedged** P&L separately — currency can dominate equity beta over short horizons.
- **Brazilian access (acesso a partir do Brasil):**
  - **Regional ETFs (most practical):** **Global X FTSE Southeast Asia ETF (ASEA)** tracks the 40 largest liquid names across Singapore, Malaysia, Indonesia, Thailand and the Philippines <https://money.usnews.com/funds/etfs/pacific-asia-ex-japan-stk/global-x-ftse-southeast-asia-etf/asea>; **iShares MSCI Emerging Markets Asia ETF (EEMA)** for broader EM-Asia <https://www.ishares.com/us/products/239629/ishares-msci-emerging-markets-asia-etf>; **VanEck Vietnam ETF (VNM)** for single-country Vietnam exposure.
  - **Single-country ETFs (US-listed):** **EWS** (Singapore), **EIDO** (Indonesia), **THD** (Thailand), **EWM** (Malaysia), **EPHE** (Philippines), **VNM** (Vietnam).
  - **B3 / BDRs:** Brazilians can buy **BDRs** (Brazilian Depositary Receipts) of US-listed Asia/EM ETFs and some ADRs on **B3**; pure ASEAN single-stock BDRs are rare, so EM/Asia ETF-BDRs and direct offshore brokerage (via *conta no exterior*) are the usual routes. Direct local accounts in IDX/SET/HOSE are impractical for retail foreigners.

---

## 9. Data & APIs (libraries, endpoints, ticker conventions)

| Source | Coverage | Access | Ticker / suffix convention |
|---|---|---|---|
| **yfinance** (Python) | SGX/IDX/SET/Bursa/PSE quotes & history | free, Yahoo-backed | `D05.SI` (DBS), `BBCA.JK`, `PTT.BK`, `1155.KL` (Maybank), `SM.PS` |
| **vnstock** (Python) | HOSE/HNX/UPCoM equities, VN-Index, VN30, financials, funds, derivatives | free OSS, TCBS/SSI backends | bare symbols, e.g. `VIC`, `VNM`, `HPG`; index `VNINDEX` |
| **SGX APIs / website** | SGX cash + derivatives reference data, indices | official | native SGX codes (e.g. `D05`, `STI`) |
| **SET / SETSMART** | SET historical, fundamentals, index constituents | official (paid SETSMART; free site data) | SET symbol (e.g. `PTT`, `AOT`) |
| **IDX website / KSEI** | IDX listings, IHSG, corporate actions | official | 4-letter codes (e.g. `BBCA`, `TLKM`) |
| **Bursa Malaysia** | KLCI, listings, derivatives | official | stock number/code (e.g. `1155` Maybank) |
| **PSE Edge** | PSE disclosures, PSEi constituents | official | symbol (e.g. `SM`, `ALI`, `BDO`) |
| **FTSE Russell / MSCI** | STI, FBM KLCI, MSCI ASEAN index methodology & constituents | index providers | index codes |

**Ticker/suffix cheat-sheet (yfinance):**
`.SI` Singapore · `.JK` Jakarta/Indonesia · `.BK` Bangkok/Thailand · `.KL` Kuala Lumpur/Malaysia · `.PS` Philippines. Vietnam is the exception — there is no reliable Yahoo suffix; use `vnstock`.

**vnstock quick reference:** PyPI `pip install vnstock`; repos <https://github.com/thinh-vu/vnstock> and the official org <https://github.com/vnstock-official/vnstock>; an API wrapper at <https://github.com/vnstock-hq/vnstock-api>. It exposes equity history, financial statements, VNINDEX/HNX/UPCoM index series, warrants, futures, funds and macro/commodities.

---

## 10. ML / Quant angle

- **Price-limit-aware modelling:** every ASEAN market enforces **daily price bands** on individual names (Thailand ±30%, Indonesia ±15% ARB, Philippines ±50% static, Vietnam ±7%/±20% first day, with first-day IPO multiples in SET). Returns are **censored/truncated** at limit-up/limit-down — naive Gaussian return models and stop-loss backtests break here. Use **limit-hit indicators** as features, model the **probability of touching the band**, and treat limit days as missing/right-censored observations.
- **Liquidity & microstructure:** thin order books, wide tiered tick sizes (SET) and lunch-break sessions create **intraday seasonality** and overnight gaps — important for execution models, VWAP/TWAP scheduling and slippage estimation.
- **Index-event / reconstitution arbitrage:** semi-annual **SET50/SET100**, quarterly **STI**, and FTSE/MSCI ASEAN reviews drive predictable add/delete flows — a classic event-driven ML target (predict additions from cap/free-float/liquidity screens).
- **Frontier→EM reclassification:** **Vietnam's** FTSE upgrade path and "foreign room" constraints are a structural, forecastable flow catalyst; foreign-ownership-limit data is itself an alpha feature.
- **Cross-asset / FX overlay:** because P&L is multi-currency, quant strategies must **jointly model equity and local-FX** (carry, capital controls in IDR/VND). Currency regimes can flip the sign of USD returns versus local-index returns.
- **Data hygiene:** corporate actions, lot changes (SET 100→50 reduction at THB 500), board reclassifications (IDX boards) and ticker reuse demand careful point-in-time (PIT) datasets to avoid survivorship/look-ahead bias.

---

## Sources

- SGX / STI: <https://www.sgx.com/> · <https://www.sgx.com/indices/products/sti> · <https://www.sgx.com/derivatives/products/chinaa50> · <https://www.lseg.com/content/dam/ftse-russell/en_us/documents/ground-rules/straits-times-index-ground-rules.pdf> · <https://capitalmarkets.sg/guides/sgx-trading-explained.php>
- GIFT Nifty / SGX Nifty migration: <https://en.wikipedia.org/wiki/GIFT_Nifty> · <https://www.sgx.com/derivatives/products/gift-connect>
- IDX: <https://en.wikipedia.org/wiki/Indonesia_Stock_Exchange> · <https://www.idx.co.id/en/news/tplus2-settlement/> · <https://aei.or.id/en/press-release/understanding-auto-rejection-in-the-indonesia-stock-exchange-mechanisms-and-post-pandemic-adjustments>
- SET: <https://www.set.or.th/en/market/information/trading-procedure/trading-hours> · <https://www.set.or.th/en/trading-units-tick-sizes-price-limits> · <https://www.set.or.th/en/market/index/set50/profile> · <https://mondovisione.com/media-and-resources/news/set-to-implement-temporary-measures-on-ceiling-and-floor-limits-dynamic-price-ban-202547/> · <https://mondovisione.com/media-and-resources/news/set-to-restore-normal-trading-rules-on-ceiling-and-floor-limits-and-dynamic-pri-2025624/>
- Bursa Malaysia: <https://www.bursamalaysia.com/trade/our_products_services/indices/ftse_bursa_malaysia_indices/ftse_bursa_malaysia_klci> · <https://www.bursamalaysia.com/trade/post_trade/securities-clearing-and-settlement/overview> · <https://www.bursamalaysia.com/regulation/securities/rules_of_bursa_malaysia_securities>
- PSE: <https://www.pse.com.ph/> · <https://www.pse.com.ph/investing-at-pse/> · <https://documents.pse.com.ph/wp-content/uploads/sites/15/2021/04/Revised-Trading-Rules.pdf> · <https://www.pse.com.ph/sccp-marks-successful-migration-to-the-t2-settlement-cycle/> · <https://www.tradinghours.com/markets/pse>
- Vietnam: <https://en.wikipedia.org/wiki/Ho_Chi_Minh_City_Stock_Exchange> · <https://www.regulationasia.com/vietnams-krx-trading-system-goes-live> · <https://github.com/thinh-vu/vnstock> · <https://github.com/vnstock-official/vnstock>
- ASEAN-wide / ETFs: <https://www.aseanexchanges.org/> · <https://money.usnews.com/funds/etfs/pacific-asia-ex-japan-stk/global-x-ftse-southeast-asia-etf/asea> · <https://www.ishares.com/us/products/239629/ishares-msci-emerging-markets-asia-etf>

**Keywords:** ASEAN equities, Southeast Asia stock markets, SGX Singapore Exchange, Straits Times Index STI, IDX Bursa Efek Indonesia IHSG, SET Stock Exchange of Thailand SET50, Bursa Malaysia FTSE KLCI, PSE PSEi, HOSE HNX VN-Index VN30, price limit auto rejection, settlement T+2, board lot, GIFT Nifty, FTSE China A50, vnstock, yfinance suffixes, BDR, ETF EM Asia · *mercados asiáticos, bolsa do Sudeste Asiático, índice de ações, limite de oscilação (circuit breaker), liquidação T+2, lote-padrão, BDR Brasil B3, ETF mercados emergentes, dados de mercado, quant e aprendizado de máquina.*
