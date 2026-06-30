# Commodities & Energy Data Sources

> A field guide to the **fundamental, positioning, flow, weather and satellite data** that actually drives commodity & energy ML — most of it free from government agencies, the rest from specialized vendors. Complements the generic market-data pages: here the alpha lives in *supply-demand balances, inventories, vessel flows and weather*, not just price ticks.

This page focuses on **commodity- and energy-specific** sources (oil, gas, power, metals, ags, softs) that are **not** in the generic data-API or equity pages. For futures mechanics, roll/continuation bias, and trend/carry strategy theory see [`./README.md`](./README.md). For price-tick vendors (Polygon, ORATS, Deribit, FRED basics) see [`../Algorithmic_and_Quant_Trading/`](../Algorithmic_and_Quant_Trading/) and [`../Alternative_Data_and_Sentiment_Analysis/`](../Alternative_Data_and_Sentiment_Analysis/).

---

## 1. Why commodity ML needs different data

Equity ML lives on prices, fundamentals and text. Commodity/energy ML lives on **physical reality**: how many barrels are in storage, how many tankers are at sea, how much corn the USDA forecasts, how cold the next two weeks will be, and *who is positioned which way*. The highest-Sharpe commodity signals historically come from **inventories, term-structure/carry, positioning (COT) and weather** — not from the price series alone. Three hard caveats run through everything below:

- **Point-in-time / vintage**: USDA WASDE and EIA weekly numbers are *revised*. Backtesting on the latest-vintage table is a classic **look-ahead bias**. Use the as-released figure (archived PDFs/CSVs) for the release date.
- **Release-timing edge**: EIA WPSR (Wed 10:30 ET), EIA gas storage (Thu 10:30 ET), USDA WASDE/Crop Reports (noon ET), CFTC COT (Fri 15:30 ET), Baker Hughes rigs (Fri noon CT). These are *scheduled market-moving events*; align features to the exact timestamp.
- **Continuous-contract construction**: free continuous series (CHRIS) silently miss/corrupt sections. See `./README.md` roll-bias notes.

---

## 2. Energy — oil, gas, power (government / free)

| Source | Commodity / content | Coverage | Free/Paid | API / how | Link |
|---|---|---|---|---|---|
| **EIA Open Data API v2** | Petroleum, natural gas, electricity, coal, renewables, nuclear, CO₂, prices, storage, production | US (some intl); daily→annual; 1920s→ for some series | **Free** (free key) | RESTful, hierarchical: `https://api.eia.gov/v2/` + `?api_key=`; bulk download (no key); 2×/day update | https://www.eia.gov/opendata/ |
| **EIA Weekly Petroleum Status (WPSR)** | Crude/products stocks, refinery runs, imports, production | US weekly, Wed 10:30 ET | **Free** | CSV/XLS + PDF at release site; also in API v2 | https://ir.eia.gov/wpsr/wpsrsummary.pdf |
| **EIA Weekly Natural Gas Storage (WNGSR)** | Working gas in underground storage by region | US weekly, Thu 10:30 ET | **Free** | CSV at ir.eia.gov; in API v2 | https://ir.eia.gov/ngs/ngs.html |
| **JODI-Oil / JODI-Gas** | Production, demand, imports/exports, stocks (national submissions) | ~100 countries (~90% of oil S&D), monthly, 2002→ | **Free** | Full DB download (.ivt/.csv); Beyond 20/20 web DB | https://www.jodidata.org/oil/database/data-downloads.aspx |
| **IEA Oil Market Report (OMR)** | Supply, demand, inventories, refining, OECD stocks | Global, monthly | Headline **free** (full data **paid**, public 3 months delayed) | PDF + paid data products (MODS) | https://www.iea.org/data-and-statistics/data-product/oil-market-report-omr |
| **OPEC Monthly Oil Market Report (MOMR)** | OPEC+ production, world balances, secondary-source crude output | Global, monthly | **Free** | PDF / Excel tables | https://www.opec.org/monthly-oil-market-report.html |
| **Baker Hughes Rig Count** | Active oil/gas drilling rigs (leading supply indicator) | North America weekly (last work-day, noon CT); Intl monthly | **Free** | Excel/CSV download; also via ICE Connect | https://rigcount.bakerhughes.com/ |
| **ENTSO-E Transparency Platform** | EU electricity: load, generation by source, day-ahead prices, cross-border flows, outages, balancing | Pan-European, since 2015 | **Free** (free API token) | RESTful API (email to enable) + SFTP bulk CSV; Python `entsoe-py` | https://transparency.entsoe.eu/ |
| **FRED (energy series)** | WTI, Brent, Henry Hub, gold, etc. (St. Louis Fed mirror) | Long history | **Free** | API/CSV (covered in FRED-basic elsewhere) | https://fred.stlouisfed.org/ |

> Python: **`myeia`** (EIA v2 wrapper, https://github.com/philsv/myeia), legacy `EIA-python` (https://github.com/mra1385/EIA-python). EIA's API v2 hit v2.1.10 in Oct 2025 with more consistent HTTP status codes.

---

## 3. Positioning — CFTC Commitments of Traders (COT)

The COT report is the single most-used **positioning** dataset in systematic commodity trading (net spec/commercial positions → crowding & reversal signals). Released Friday 15:30 ET for Tuesday's data.

| Report | What it splits | History | Access |
|---|---|---|---|
| **Legacy** (Futures-only / Combined) | Commercial vs. non-commercial vs. non-reportable | 1986→ (weekly since 2000) | Free |
| **Disaggregated** | Producer/merchant, **Swap Dealers**, **Managed Money**, Other reportable | 2006-06-13→ | Free |
| **Traders in Financial Futures (TFF)** | Dealer, Asset Manager, Leveraged Funds, Other | 2006→ | Free |
| **Supplemental (CIT)** | Index traders in select ag markets | — | Free |

**APIs / how:**
- **Socrata public API** (no auth required; JSON; stable). Dataset IDs e.g. Legacy-Futures `6dca-aqww`, Disaggregated-Futures `72hh-3qpy`, TFF-Futures `gpe5-46if`, Disaggregated-Combined `kh3c-gbw2`. Browse: https://publicreporting.cftc.gov/ · API foundry: https://dev.socrata.com/foundry/publicreporting.cftc.gov/6dca-aqww
- App token optional (un-throttled if used); register at https://publicreporting.cftc.gov/profile/edit/developer_settings
- Python: **`cot_reports`** (https://github.com/NDelventhal/cot_reports), **`cftc-cot`** (https://github.com/Mcamin/cftc-cot), `openbb-cftc`.
- Free dashboards: [Barchart COT](https://www.barchart.com/futures/commitment-of-traders), [cot-reports.com](https://cot-reports.com/), CME QuikStrike.

---

## 4. Agriculture & softs (government / free)

| Source | Commodity / content | Coverage | Free/Paid | API / how | Link |
|---|---|---|---|---|---|
| **USDA NASS Quick Stats** | US production, yield, acreage, stocks, prices (Census + survey) | US, county→national, 1866→ | **Free** (free key) | REST API + bulk `.gz`; key at quickstats.nass.usda.gov/api | https://quickstats.nass.usda.gov/api |
| **USDA WASDE** | World Agricultural Supply & Demand Estimates (balance sheets) | Global, monthly (noon ET) | **Free** | PDF/XML/Excel/Text per release + "Consolidated Historical WASDE Report Data" CSV (2010→) | https://www.usda.gov/about-usda/general-information/staff-offices/office-chief-economist/commodity-markets/wasde-report |
| **USDA NASS Crop Production / Crop Progress** | Forecasts, condition ratings, harvest progress | US weekly/monthly | **Free** | In Quick Stats + PDF | https://www.nass.usda.gov/ |
| **USDA FAS PSD Online** | Production, Supply & Distribution by country/commodity | Global | **Free** | Web API (key via api.data.gov) | https://apps.fas.usda.gov/psdonline/ |
| **USDA FAS Export Sales (ESR) / GATS** | Weekly US export sales & shipments; trade flows | US/global | **Free** | ESRQS portal (new since 2026) + OpenData web API (key via api.data.gov): apps.fas.usda.gov/opendatawebV2 | https://www.fas.usda.gov/data |
| **UN Comtrade** | International trade by HS/SITC commodity code | Global, monthly/annual, 1962→ | **Free** (tiered) | REST API + bulk | https://comtradeplus.un.org/ |

> NASS / WASDE are **revised** — archive each release for point-in-time backtests. Python: `rnassqs` (R), USDA FAS samples on data.gov.

---

## 5. Metals & precious metals

| Source | Content | Coverage | Free/Paid | API / how | Link |
|---|---|---|---|---|---|
| **LME (London Metal Exchange)** | Official/closing prices, stocks (on/off-warrant), 3-month, spreads — Cu, Al, Zn, Ni, Pb, Sn, Co | Global benchmark | **Paid** (delayed/historical; LMElive app) | LMEsource/LMEselectMD feeds; next-day delayed XML feed (~$2,490/yr); buy historical via LME Portal Data Services; redistributed via ICE/LSEG | https://www.lme.com/market-data |
| **LBMA Precious Metal Prices** | Gold AM/PM, Silver, Platinum, Palladium auction benchmarks | Global benchmark | **Free** (non-commercial) via MyLBMA; license for commercial | MyLBMA portal; CSV | https://www.lbma.org.uk/prices-and-data/lbma-precious-metal-prices |
| **FRED / Nasdaq Data Link** | Long-history gold/silver fixings | Global | **Free** | API/CSV | https://fred.stlouisfed.org/ |
| **COMEX (CME COMEX)** | Gold/silver/copper futures, warehouse stocks | US | Quotes free (delayed); deep data paid | CME DataMine / API | https://www.cmegroup.com/markets/metals.html |

> COMEX/LME **warehouse stock** changes are a fundamental signal akin to oil inventories.

---

## 6. Exchange & continuous-futures data (price + curve)

| Source | Content | Free/Paid | API / how | Link |
|---|---|---|---|---|
| **CME DataMine** | CME/CBOT/NYMEX/COMEX historical futures & options, BrokerTec/EBS cash | **Paid** (self-service, monthly/annual) | Data API, SFTP, S3, file browser | https://www.cmegroup.com/datamine.html |
| **ICE Data Services** | ICE futures (Brent, gasoil, softs, power, gas), ref data | **Paid** | ICE Connect + APIs; also hosts Baker Hughes, Kpler, LME | https://developer.ice.com/ |
| **Nasdaq Data Link (Quandl)** | CHRIS Wiki Continuous Futures (free), **SCF** Stevens Continuous (paid) | Mixed — CHRIS **free** w/ key; SCF ~$50/mo | REST API; Python `nasdaq-data-link` | https://data.nasdaq.com/data/CHRIS-wiki-continuous-futures/documentation/introduction |
| **B3 / 🇧🇷** | Brazilian futures (boi gordo, café, milho, etanol, DI, Ibov), market data | Quotes free (delayed); UP2DATA paid | B3 UP2DATA feeds | https://www.b3.com.br/ |

> **CHRIS** (free) is fine for prototyping but can miss/corrupt sections → use **SCF** or exchange data for production. See `./README.md` for back-adjustment/roll bias.

---

## 7. Premium energy/commodity intelligence (Price Reporting Agencies & analytics)

These are the **paid** institutional standards — benchmark assessments and fundamental analytics. Names matter even if you can't license them (their assessments *are* the contract pricing).

| Vendor | Content | Free/Paid | Notes / API |
|---|---|---|---|
| **S&P Global Commodity Insights (Platts)** | 12,000+ daily price assessments: crude, products, gas/power, LNG, chemicals, metals, ags, shipping | **Paid** | Platts Market Data feeds, Dimensions Pro | https://www.spglobal.com/commodity-insights/ |
| **Argus Media** | PRA assessments across crude/products, gas/LNG, power, metals, fertilizers, biofuels | **Paid** | Argus Direct / data feeds | https://www.argusmedia.com/ |
| **Wood Mackenzie** | Upstream/downstream, power & renewables, metals & mining models, cost curves | **Paid** | Lens platform / Lens Direct API (OData) | https://www.woodmac.com/ |
| **ICIS** | Chemicals, energy, fertilizers price reporting & analytics | **Paid** | data feeds | https://www.icis.com/ |
| **LSEG (Refinitiv) Workspace / Datastream** | Commodities curves, fundamentals, news; mine production & cost curves | **Paid** | **Eikon retired 30 Jun 2025** → use **LSEG Data Library for Python** / Datastream Web Service | https://developers.lseg.com/ |
| **Bloomberg** | BCOM, commodity curves, COT, fundamentals | **Paid** | BLPAPI / `blpapi` | https://www.bloomberg.com/ |

---

## 8. Flow & alternative data — tankers, satellites, weather

The real edge for many funds. **Vessel flows** nowcast supply; **satellite tank/crop** imagery nowcasts inventory/yield; **weather** drives gas & ag demand.

### 8a. Shipping / cargo flows (mostly paid)

| Vendor | Content | Free/Paid | API | Link |
|---|---|---|---|---|
| **Kpler** | Seaborne flows for 40+ commodities (crude, products, LNG, LPG, coal, ags, metals), storage, 10,000+ ports, 15,000+ assets | **Paid** (modular) | Real-time REST API | https://www.kpler.com/ |
| **Vortexa** | Crude/products/LNG cargo flows, floating storage, freight | **Paid** | API | https://www.vortexa.com/ |
| **Kpler AIS (ex-MarineTraffic / FleetMon / Spire)** | Raw AIS vessel tracking (terrestrial + satellite), port calls, ETAs. Kpler acquired MarineTraffic+FleetMon (2023) and Spire Maritime (2025), then unified them as **"Kpler AIS"** launched **19 Sep 2025** | **Paid** | AIS API; MarineTraffic.com front-end | https://www.kpler.com/product/maritime/data-services |
| **Datalastic / VesselAPI** | Lower-cost AIS REST APIs (indie alternatives; plans from ~€99–199/mo) | **Paid** (cheaper tiers) | REST | https://datalastic.com/ |

### 8b. Satellite / geospatial (paid; some free imagery)

| Vendor | Content | Free/Paid | Link |
|---|---|---|---|
| **Orbital Insight (now Privateer)** | Crude oil **floating-roof tank storage** (shadow analysis, ~25k tanks), retail/foot-traffic | **Paid** | https://en.wikipedia.org/wiki/Orbital_Insight |
| **Planet Labs** | Daily PlanetScope + high-res SkySat imagery (crop/oil monitoring source) | **Paid** (some free/research) | https://www.planet.com/ |
| **Descartes Labs** (now part of EarthDaily Analytics, acq. Oct 2024) | Crop-yield forecasting, refinery/energy geospatial analytics (products: Marigold, Iris, Ascend) | **Paid** | https://www.descarteslabs.com/ |
| **Sentinel-2 / Copernicus, NASA Landsat/MODIS** | Free raw optical imagery — DIY crop-NDVI / storage models | **Free** | https://dataspace.copernicus.eu/ |
| **Google Earth Engine** | Petabyte catalog (Sentinel, Landsat, ERA5, MODIS) + compute | **Free** for non-commercial/research (commercial/operational use now paid) | https://earthengine.google.com/ |

### 8c. Weather & climate (free + paid)

| Source | Content | Coverage | Free/Paid | API | Link |
|---|---|---|---|---|---|
| **NOAA NCEI Climate Data Online (CDO) / GHCN-Daily** | Tmax/Tmin, precip, snow from 100k+ stations, 180 countries | Global, daily | **Free** (free token) | REST v2 + `access/services/data/v1`; token at ncdc.noaa.gov/cdo-web/token | https://www.ncei.noaa.gov/cdo-web/ |
| **NOAA CPC / GFS / NWS** | HDD/CDD, 6-10 & 8-14 day outlooks, GFS forecasts | US/global | **Free** | grib/CSV/API | https://www.cpc.ncep.noaa.gov/ |
| **Copernicus CDS — ERA5** | Hourly reanalysis (temp, wind, solar, precip), ~31 km, 1940→ | Global | **Free** | `cdsapi` Python / `ecmwf-datastores-client` | https://cds.climate.copernicus.eu/ |
| **ECMWF (operational forecasts)** | High-skill medium-range NWP | Global | **Paid** (open data subset free) | API | https://www.ecmwf.int/ |
| **Open-Meteo** | Free historical + forecast weather REST (great for prototyping) | Global | **Free** | REST, no key | https://open-meteo.com/ |

> Weather → **HDD/CDD** features for natural-gas & power demand; **growing-degree-days / NDVI** for crop yield. ERA5 is the standard free reanalysis baseline.

---

## 9. 🇧🇷 Brazil — commodity & energy data

Brazil is a top global producer (soy, corn, coffee, sugar, beef, iron ore, oil) — these are the authoritative local sources, mostly free.

| Source | Commodity / content | Free/Paid | API / how | Link |
|---|---|---|---|---|
| **CEPEA-Esalq/USP** | Reference indicators (*indicadores*): boi gordo, soja, milho, café, açúcar, etanol, algodão, leite, suíno, frango, trigo... + IPPA producer-price index | **Free** to view (site, since 2001); **paid** bulk API (complete module ~R$10,500/yr) | Web tables / widget; paid `onetoone.cepea.org.br/api` | https://www.cepea.org.br/br |
| **CONAB** | Brazilian crop surveys (*safras*), production/area/yield forecasts, stocks, PGPM prices | **Free** | Portal + open data / dashboards | https://www.conab.gov.br/ |
| **ANP** (oil, gas, biofuels) | Production by well, reserves, fuel **price survey** (gasoline/diesel/ethanol), imports/exports, RenovaBio | **Free** | Dados Abertos CSV + Painéis Dinâmicos (Power BI); dados.gov.br | https://www.gov.br/anp/pt-br/centrais-de-conteudo/dados-abertos |
| **ONS** (grid operator) | Generation by source, load, hydro reservoir levels, transmission, reliability | **Free** | Portal Dados Abertos (datasets + API) | https://dados.ons.org.br/ |
| **CCEE** (power market) | **PLD** (settlement price) historical since 2001, free-market (*mercado livre*) volumes, contracts | **Free** | Dados Abertos CCEE (CKAN datasets + API): `pld_horario`, `pld_media_diaria`, `pld_media_mensal` (+ weekly), expanded to 4 granularities in 2025 | https://dadosabertos.ccee.org.br/ |
| **ANEEL** | Tariffs, generation registry, distribution data | **Free** | Portal Dados Abertos | https://dadosabertos.aneel.gov.br/ |
| **B3** | Agro & financial futures (boi, café, milho, etanol hidratado, soja, ouro) | Delayed free / UP2DATA paid | feeds | https://www.b3.com.br/ |

> 🇧🇷 ML angles: ONS reservoir levels + ENA (*energia natural afluente*) → **PLD** nowcasting; ANP fuel-price survey → margin/arbitrage; CONAB/CEPEA + Copernicus weather → soy/corn yield & basis models. CCEE PLD is the Brazilian analogue of US power LMP and a prime RL/forecasting target.

---

## 10. Putting it together (ML workflow patterns)

| Signal family | Primary data | Free? | Typical model |
|---|---|---|---|
| **Inventory nowcast** (oil/gas) | EIA WPSR/WNGSR, JODI, satellite tanks, Kpler floating storage | mixed | GBDT / regression on surprise vs. consensus |
| **Crop yield / balance** | USDA NASS+WASDE+PSD, CONAB, NDVI (Sentinel), ERA5/GDD | mostly free | CNN on imagery + tabular GBDT |
| **Positioning / crowding** | CFTC COT (Disaggregated, TFF) | free | Net-position z-scores, reversal features |
| **Demand (gas/power)** | NOAA HDD/CDD, ERA5, ONS load, ENTSO-E, EIA | free | Weather-normalized demand regressions |
| **Flow / supply** | Kpler, Vortexa, AIS, Baker Hughes rigs | mixed | Lead-lag flow → price |
| **Term-structure / carry** | CME DataMine, Nasdaq DL continuous, ICE | mixed | Curve features → TSMOM/carry |

**Bias checklist (commodities-specific):** ✅ point-in-time vintages for revised gov't stats · ✅ exact release timestamps (event alignment) · ✅ robust continuous-contract roll (avoid CHRIS gaps) · ✅ survivorship in delisted/illiquid contracts · ✅ license terms before redistributing PRA/exchange data.

---

## Sources

- EIA Open Data: https://www.eia.gov/opendata/ · API docs: https://www.eia.gov/opendata/documentation.php · WPSR: https://ir.eia.gov/wpsr/wpsrsummary.pdf · WNGSR: https://ir.eia.gov/ngs/ngs.html
- JODI: https://www.jodidata.org/oil/database/data-downloads.aspx · IEA OMR: https://www.iea.org/data-and-statistics/data-product/oil-market-report-omr · OPEC MOMR: https://www.opec.org/monthly-oil-market-report.html
- Baker Hughes: https://rigcount.bakerhughes.com/ · ENTSO-E Transparency Platform: https://transparency.entsoe.eu/
- CFTC COT: https://publicreporting.cftc.gov/ · Socrata: https://dev.socrata.com/foundry/publicreporting.cftc.gov/6dca-aqww · `cot_reports`: https://github.com/NDelventhal/cot_reports
- USDA NASS Quick Stats: https://quickstats.nass.usda.gov/api · WASDE: https://www.usda.gov/about-usda/general-information/staff-offices/office-chief-economist/commodity-markets/wasde-report · FAS PSD: https://apps.fas.usda.gov/psdonline/ · FAS data: https://www.fas.usda.gov/data · FAS OpenData: https://apps.fas.usda.gov/opendatawebV2/ · UN Comtrade: https://comtradeplus.un.org/
- LME: https://www.lme.com/market-data · LBMA: https://www.lbma.org.uk/prices-and-data/lbma-precious-metal-prices
- CME DataMine: https://www.cmegroup.com/datamine.html · ICE Developer: https://developer.ice.com/ · Nasdaq Data Link CHRIS: https://data.nasdaq.com/data/CHRIS-wiki-continuous-futures/documentation/introduction
- S&P Global Commodity Insights: https://www.spglobal.com/commodity-insights/ · Argus: https://www.argusmedia.com/ · Wood Mackenzie: https://www.woodmac.com/ · ICIS: https://www.icis.com/ · LSEG Developers: https://developers.lseg.com/
- Kpler: https://www.kpler.com/ · Vortexa: https://www.vortexa.com/ · Kpler AIS (ex-MarineTraffic/FleetMon/Spire): https://www.kpler.com/product/maritime/data-services · Datalastic: https://datalastic.com/
- Orbital Insight: https://en.wikipedia.org/wiki/Orbital_Insight · Planet: https://www.planet.com/ · Descartes Labs: https://www.descarteslabs.com/ · Copernicus: https://dataspace.copernicus.eu/ · Earth Engine: https://earthengine.google.com/
- NOAA CDO/GHCN: https://www.ncei.noaa.gov/cdo-web/ · Copernicus CDS ERA5: https://cds.climate.copernicus.eu/ · Open-Meteo: https://open-meteo.com/
- 🇧🇷 CEPEA: https://www.cepea.org.br/br · CONAB: https://www.conab.gov.br/ · ANP: https://www.gov.br/anp/pt-br/centrais-de-conteudo/dados-abertos · ONS: https://dados.ons.org.br/ · CCEE: https://dadosabertos.ccee.org.br/ · ANEEL: https://dadosabertos.aneel.gov.br/

**Keywords:** commodity data, energy data, EIA API, oil inventories, natural gas storage, CFTC COT (posicionamento), Commitments of Traders, USDA WASDE/NASS, crop yield (safra), Baker Hughes rig count, LME metals, LBMA gold, CME DataMine, continuous futures (contratos contínuos), Kpler Vortexa tanker flows (fluxo de navios), Kpler AIS MarineTraffic, satellite oil storage / crop (satélite), NOAA GHCN weather (clima), Copernicus ERA5, HDD CDD, ENTSO-E European power, price reporting agency (Platts Argus), 🇧🇷 CEPEA ESALQ agro, ANP petróleo gás, ONS CCEE PLD energia, point-in-time, look-ahead bias, alternative data (dados alternativos).
