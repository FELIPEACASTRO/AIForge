# B3 Options & Derivatives (Opções e Derivativos no Brasil)

> A dense reference to Brazil's listed options and futures market — equity options, index/FX/rates/agro futures, exact contract specs, day-trade mechanics, clearing, COE, plus the ML/quant and data-API stack used to model it.

**B3 S.A. – Brasil, Bolsa, Balcão** is the sole securities, commodities and derivatives exchange in Brazil, formed by the 2017 merger of BM&FBOVESPA and CETIP. It is one of the largest exchange groups in the world by listed-derivatives volume, and acts as its own **central counterparty (CCP)** and clearinghouse. This page focuses on the listed/registered **derivatives** complex — options (opções) and futures (futuros) — that quants, traders and investors actually trade.

> Disclaimer: contract specs change. Always confirm against the official B3 contract page before trading. Values below are sourced to B3 (mid-2025/2026) and cited inline.

---

## 1. Equity Options (Opções sobre Ações)

Listed options on individual stocks (e.g. PETR4, VALE3, BBAS3) are the most retail-accessible derivative on B3. Each **series (série)** is identified by a code combining the underlying ticker, a **series letter (letra da série)** encoding type + expiration month, and a strike reference ([B3 Educação](https://edu.b3.com.br/w/opcoes-acoes)).

### Ticker / letter convention (convenção de código)

The standard option code is **4 letters (underlying) + 1 series letter + strike number(s)**, e.g. `PETRA34`. The 5th character encodes both **call vs put** and the **expiration month** ([ADVFN](https://br.advfn.com/investimentos/opcoes/codigo-bovespa), [B3 settlement-code PDF](https://www.b3.com.br/data/files/4B/F3/C4/10/519AC7109A21A9C78C094EA8/Formacao%20do%20Codigo%20de%20Liquidacao%20das%20Opcoes.pdf)):

| Month | Call letter (compra) | Put letter (venda) |
|-------|----------------------|--------------------|
| January (Jan) | A | M |
| February (Fev) | B | N |
| March (Mar) | C | O |
| April (Abr) | D | P |
| May (Mai) | E | Q |
| June (Jun) | F | R |
| July (Jul) | G | S |
| August (Ago) | H | T |
| September (Set) | I | U |
| October (Out) | J | V |
| November (Nov) | K | W |
| December (Dez) | L | X |

So **calls use A–L** and **puts use M–X** by month. The trailing number is the strike-price reference (preço de exercício). The letter itself does **not** reveal American vs European style ([Melver](https://www.melver.com.br/blog/entenda-os-codigos-das-opcoes-original/)).

### Style, expiration & exercise

- **Exercise style (estilo):** B3 lists both **American** (americana — exercisable any business day up to expiry) and **European** (europeia — only on expiry date) series ([B3 Educação](https://edu.b3.com.br/w/opcoes-acoes)).
- **Monthly expiration (vencimento):** standard equity-option series expire on the **third Friday (terceira sexta-feira)** of the contract month ([B3 — calendário de vencimentos de opções](https://www.b3.com.br/pt_br/solucoes/plataformas/puma-trading-system/para-participantes-e-traders/calendario-de-negociacao/vencimentos/calendario-de-vencimentos-de-opcoes-sobre-acoes-e-indices/)).
- **Weekly options (opções semanais):** B3 also lists weekly equity options, expiring every Friday **except** the third Friday (which is the monthly expiry) ([B3 — opções semanais](https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/renda-variavel/opcoes-semanais-sobre-acoes.htm)).
- **Settlement on exercise:** physical-style — the holder (titular) buys/sells the underlying at the strike and the writer (lançador) takes the opposite side ([B3 Educação](https://edu.b3.com.br/w/opcoes-acoes)).

### Options on the Ibovespa (Opções sobre Ibovespa)

B3 also lists **index options on the Ibovespa**, cash-settled standardized contracts that let investors trade a view on the broad Brazilian equity market without single-name risk ([B3 — Options on Ibovespa (EN)](https://www.b3.com.br/en_us/products-and-services/trading/equities/options-on-ibovespa.htm)).

---

## 2. The Big Futures Contracts (Os principais contratos futuros)

These are the high-volume futures that dominate B3 derivatives. The **mini contracts (minicontratos)** — WIN and WDO — drive Brazil's retail day-trade culture.

| Contract | Code (código) | Underlying (objeto) | Contract size / point value | Min. tick | Expiration | Settlement |
|----------|---------------|---------------------|------------------------------|-----------|------------|------------|
| Ibovespa Futures (cheio) | **IND** | Ibovespa index | **R$1.00 / index point** | 5 pts | Even months, Wed nearest the 15th | Cash |
| Mini Ibovespa Futures | **WIN** | Ibovespa index | **R$0.20 / index point** | 5 pts | Even months, Wed nearest the 15th | Cash |
| U.S. Dollar Futures (cheio) | **DOL** | BRL/USD comercial | **US$50,000** notional | R$0.5 per US$1,000 | 1st business day of month | Cash |
| Mini Dollar Futures | **WDO** | BRL/USD comercial | **US$10,000** notional | R$0.5 per US$1,000 | 1st business day of month | Cash |
| One-day Interbank Deposit (DI) | **DI1** | DI / interest rate | **PU = R$100,000** at maturity | 0.001/0.005 rate pt | 1st business day of month | Cash |

Sources: [Mini Ibovespa Futures – B3 (EN)](https://www.b3.com.br/en_us/products-and-services/trading/equities/mini-ibovespa-futures.htm), [Ibovespa Futures – B3 (EN)](https://www.b3.com.br/en_us/products-and-services/trading/equities/cash-equities/ibovespa-futures.htm), [U.S. Dollar Futures – B3 (EN)](https://www.b3.com.br/en_us/products-and-services/trading/exchange-rates/u-s-dollar-futures.htm), [Mini U.S. Dollar Futures – B3 (EN)](https://www.b3.com.br/en_us/products-and-services/trading/exchange-rates/mini-u-s-dollar-futures.htm), [One-day Interbank Deposit Futures – B3 (EN)](https://www.b3.com.br/en_us/products-and-services/trading/interest-rates/one-day-interbank-deposit-futures.htm).

### 2.1 Ibovespa futures — IND & WIN

The **mini (WIN)** is **R$0.20 per index point** vs the **full (IND)** at **R$1.00 per point**, so WIN is exactly 1/5 of IND ([B3 EN – Mini Ibovespa](https://www.b3.com.br/en_us/products-and-services/trading/equities/mini-ibovespa-futures.htm), [B3 EN – Ibovespa Futures](https://www.b3.com.br/en_us/products-and-services/trading/equities/cash-equities/ibovespa-futures.htm)). Both expire in **even months** on the **Wednesday nearest the 15th**, with a 5-point minimum variation and daily mark-to-market settlement of `(settlement − entry) × point value × contracts`. Trading hours are **09:00–18:00** (BRT), per B3.

### 2.2 Dollar futures — DOL & WDO

The **full dollar (DOL)** carries a **US$50,000** notional; the **mini (WDO)** is **US$10,000** ([B3 EN – U.S. Dollar Futures](https://www.b3.com.br/en_us/products-and-services/trading/exchange-rates/u-s-dollar-futures.htm), [B3 EN – Mini USD](https://www.b3.com.br/en_us/products-and-services/trading/exchange-rates/mini-u-s-dollar-futures.htm)). Quotation is **BRL per US$1,000**, min. tick **R$0.5 per US$1,000** (both DOL and WDO), expiring on the **1st business day of the contract month**, cash-settled (no physical delivery). Month letters follow the global futures convention: F/G/H/J/K/M/N/Q/U/V/X/Z for Jan–Dec ([Bora Investir – B3](https://borainvestir.b3.com.br/glossario/futuro-de-dolar/)).

### 2.3 DI futures — DI1 (taxa DI / juros)

The single most important rates contract in Brazil. The **DI1** quotes the **effective annual rate on a 252-business-day basis**; the traded instrument is a **unit price (PU)** that converges to **R$100,000 at maturity** (`PU = 100,000 discounted at the traded rate`) ([B3 EN – DI1](https://www.b3.com.br/en_us/products-and-services/trading/interest-rates/one-day-interbank-deposit-futures.htm)). Tick is **0.001 rate-points** (maturity ≤3 months) or **0.005** (>3 months); expiration is the **1st business day** of the contract month, cash-settled. The DI curve is the Brazilian equivalent of the swap/STIR curve and the backbone of fixed-income pricing.

### 2.4 Cupom cambial (FX coupon / DDI)

The **cupom cambial** is the USD-denominated interest rate in Brazil — effectively the local dollar rate implied by combining the DI curve and the FX forward. B3 lists **DDI futures** and **FRA de cupom** structured operations, and an explicit **structured trade combining DDI + mini-dollar (WDO)** ([B3 – DDI + WDO estruturada](https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/juros/operacao-estruturada-de-futuro-de-cupom-cambial-ddi-e-futuro-mini-de-dolar-wdo.htm)). It is central to FX-hedge and cross-currency carry strategies.

### 2.5 Agro / commodities futures (agronegócio)

Brazil is an agricultural superpower, and B3's agro complex is a global price reference for cattle, coffee and grains. The major contracts are **cash-settled** against B3/Cepea/Esalq indicators (financial settlement, no physical delivery) ([B3 Educação – Boi Gordo](https://edu.b3.com.br/w/futuro-boi-gordo)):

| Commodity | Code | Contract size | Quotation | Notes |
|-----------|------|---------------|-----------|-------|
| Live cattle (Boi Gordo) | **BGI** | 330 arrobas (@) | R$/arroba, 2 dec., tick R$0.05 | Cash-settled vs CEPEA/B3 indicator; IFBOI total-return index exists |
| Arabica coffee (Café Arábica) | **ICF** | 100 bags of 60 kg | US$/bag | Frost risk (geadas) drives sharp rallies |
| Corn (Milho) | **CCM** | 450 bags of 60 kg (27 t) | R$/bag | Cash-settled financial corn |
| Soybeans (Soja) | **SFI** | 450 bags of 60 kg (27 t) | US$/bag | Financial soy (B3 indicator) |

Sources: [B3 Educação – Boi Gordo](https://edu.b3.com.br/w/futuro-boi-gordo), [B3 – Commodities trading hours](https://www.b3.com.br/pt_br/solucoes/plataformas/puma-trading-system/para-participantes-e-traders/horario-de-negociacao/derivativos/commodities/), [B3 – calendário vencimentos agropecuários](https://www.b3.com.br/pt_br/solucoes/plataformas/puma-trading-system/para-participantes-e-traders/calendario-de-negociacao/vencimentos/calendario-de-vencimentos-de-contratos-agropecuarios/). B3 also lists gold (Ouro) and hydrous ethanol (Etanol) contracts. Verify exact sizes per contract page before trading.

---

## 3. Margin, Clearing & Day-Trade Mechanics

### Clearing & the CCP role (papel da clearing)

B3 acts as **central counterparty (CCP)** for all listed derivatives, novating every trade and guaranteeing settlement. It runs the proprietary **CORE (Close-Out Risk Evaluation)** methodology to size **margin / collateral (margem de garantia)** across an entire multi-asset portfolio, valuing the cost of closing out positions under stress ([B3 – Clearing, settlement and risk management](https://www.b3.com.br/en_us/regulation/regulatory-framework/regulations-and-manuals/clearing-settlement-and-risk-management.htm)). Margin and collateral rules are defined in the **B3 Clearinghouse Risk Management Manual**.

### Day-trade culture & leverage (alavancagem)

Brazil has an unusually large retail day-trade base built on mini contracts. Brokers post a reduced **day-trade margin (margem reduzida)** — frequently in the low hundreds of reais per mini contract (broker- and volatility-dependent) — which delivers high effective **leverage (alavancagem)** ([Bora Investir – B3 minicontratos](https://borainvestir.b3.com.br/tipos-de-investimentos/renda-variavel/day-trade/day-trade-com-mini-indice-e-minidolar-entenda-os-minicontratos/)). These intraday figures are not fixed B3 specs — they vary by broker and market regime; confirm with your corretora. Note Brazil's day-trade tax regime (20% on net gains, with 1% source withholding "dedo-duro" on day-trade results — see §7).

---

## 4. Structured Products: COE & Warrants

- **COE — Certificado de Operações Estruturadas (structured-note certificate):** a bank-issued instrument blending fixed income + a derivatives payoff (linked to indices, FX, equities, etc.), registered at B3. The popular **capital-protected (capital protegido)** variant returns at least the principal at maturity by allocating most of the proceeds to low-risk fixed income and the remainder to the embedded option; a **capital-at-risk (capital de risco)** variant offers higher upside with downside exposure ([B3 – COE](https://www.b3.com.br/pt_br/produtos-e-servicos/registro/operacoes-estruturadas/certificado-de-operacoes-estruturadas-coe.htm)). COEs must publish a DIE (Documento de Informações Essenciais) and are suitability-gated.
- **Structured operations:** B3 supports a range of registered structured operations (operações estruturadas) and rollover structures used by institutional desks.

---

## 5. ML / Quant Angle (Aprendizado de máquina e quant)

Machine learning is applied across the B3 derivatives stack:

- **Index & FX direction (WIN/WDO):** intraday classifiers and sequence models (gradient boosting, LSTMs, temporal CNNs, and increasingly transformers) on order-book/microstructure features for short-horizon direction and regime detection. The mini contracts' deep retail liquidity makes them the canonical Brazilian backtesting ground.
- **Volatility & options:** local/stochastic-volatility surfaces calibrated to listed equity and Ibovespa options; ML-augmented surface interpolation, implied-vol smoothing, and Greeks estimation; American-option pricing via Longstaff-Schwartz / neural-network LSMC.
- **Rates curve (DI1):** PCA / autoencoders on the DI term structure for curve factors (level/slope/curvature), plus state-space and ML models for SELIC-path nowcasting that drive DI1 fair value.
- **Agro:** weather/satellite features and Cepea/Esalq indicators feed seasonal models for BGI/ICF/CCM/SFI; frost-risk signals are a known fat-tail driver for ICF.
- **Execution & RL:** reinforcement-learning execution agents and market-making policies; survivorship- and look-ahead-bias control is critical given Brazil's adjustment-history quirks.
- **Risk:** portfolio-margin replication of CORE-style stress for pre-trade margin estimation; VaR/ES with EVT tails for the BRL's heavy-tailed FX behavior.

Caveat: Brazilian series demand careful handling of corporate actions, daily-adjustment (ajuste) continuity when stitching futures, holidays (B3 calendar), and the BRL's macro/regime sensitivity.

---

## 6. Data & APIs (Dados e APIs)

| Tool / source | What it gives | Notes |
|---------------|---------------|-------|
| **MetaTrader 5 (MT5) + `MetaTrader5` Python pkg** | Real-time & historical OHLCV, order routing for WIN/WDO and B3 instruments | Official MetaQuotes Python API; pulls bars into Pandas; widely used for B3 algo trading ([MQL5 docs](https://www.mql5.com/en/docs/python_metatrader5), [PyPI](https://pypi.org/project/MetaTrader5/)) |
| **Nelogica ProfitChart / Profit** | Brazilian retail trading + charting platform, DLL/API automation | Dominant local day-trade front end |
| **B3 UP2DATA / market data** | Official end-of-day & reference market-data files | B3's data distribution product |
| **B3 market-data & index pages** | Ibovespa composition, index methodology, contract specs | [Ibovespa composition](https://www.b3.com.br/pt_br/market-data-e-indices/indices/indices-amplos/indice-ibovespa-ibovespa-composicao-da-carteira.htm) |
| **brapi.dev** | Free/freemium REST API for B3 quotes, options chains, fundamentals | Brazil-focused ([brapi](https://brapi.dev/)) |
| **opcoes.net.br** | Listed B3 options chains, IV, screening | Reference for series/strikes ([opcoes.net.br](https://opcoes.net.br/opcoes/bovespa)) |
| **yfinance / investing.com** | Delayed quotes (`.SA` suffix on Yahoo) | Convenient but verify continuity |

Python quant stack typically pairs the above with `pandas`, `numpy`, `scipy`, `statsmodels`, `scikit-learn`, `QuantLib` (options/curves), `arch` (GARCH), and `vectorbt`/`backtesting.py` for research. For MT5 data extraction examples, see the [MetaTrader5 PyPI package](https://pypi.org/project/MetaTrader5/) and [MQL5 Python integration docs](https://www.mql5.com/en/docs/python_metatrader5).

---

## 7. Practical Notes (Notas práticas)

- **Liquidity hierarchy:** WIN, WDO and DI1 are the most liquid B3 derivatives; full IND/DOL are institutional; agro is liquid in nearby months only.
- **Rollover (rolagem):** index and FX futures require rolling near expiry; B3 offers structured roll operations to reduce slippage.
- **Tax:** day-trade gains are taxed at **20%** on net monthly gains, with **1% source withholding (IRRF, "dedo-duro")** that can be offset against the tax due; swing/position trades in equities, futures and options have their own regimes — consult current Receita Federal / B3 guidance ([Receita Federal – Renda variável / Bolsa de valores](https://www.gov.br/receitafederal/pt-br/assuntos/meu-imposto-de-renda/pagamento/renda-variavel/bolsa-de-valores-1)).
- **Always re-verify specs:** multipliers, ticks, margins and hours are periodically revised by B3 — the official contract page is authoritative.

---

**Sources:** [B3 Ibovespa Futures (EN)](https://www.b3.com.br/en_us/products-and-services/trading/equities/cash-equities/ibovespa-futures.htm) · [B3 Mini Ibovespa Futures (EN)](https://www.b3.com.br/en_us/products-and-services/trading/equities/mini-ibovespa-futures.htm) · [B3 U.S. Dollar Futures (EN)](https://www.b3.com.br/en_us/products-and-services/trading/exchange-rates/u-s-dollar-futures.htm) · [B3 Mini U.S. Dollar Futures (EN)](https://www.b3.com.br/en_us/products-and-services/trading/exchange-rates/mini-u-s-dollar-futures.htm) · [B3 DI1 Futures (EN)](https://www.b3.com.br/en_us/products-and-services/trading/interest-rates/one-day-interbank-deposit-futures.htm) · [B3 Opções sobre Ações (Educação)](https://edu.b3.com.br/w/opcoes-acoes) · [B3 Opções Semanais](https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/renda-variavel/opcoes-semanais-sobre-acoes.htm) · [B3 Options on Ibovespa (EN)](https://www.b3.com.br/en_us/products-and-services/trading/equities/options-on-ibovespa.htm) · [B3 Futuro Boi Gordo](https://edu.b3.com.br/w/futuro-boi-gordo) · [B3 DDI+WDO estruturada](https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/juros/operacao-estruturada-de-futuro-de-cupom-cambial-ddi-e-futuro-mini-de-dolar-wdo.htm) · [B3 COE](https://www.b3.com.br/pt_br/produtos-e-servicos/registro/operacoes-estruturadas/certificado-de-operacoes-estruturadas-coe.htm) · [B3 Clearing/Risk](https://www.b3.com.br/en_us/regulation/regulatory-framework/regulations-and-manuals/clearing-settlement-and-risk-management.htm) · [B3 settlement-code PDF](https://www.b3.com.br/data/files/4B/F3/C4/10/519AC7109A21A9C78C094EA8/Formacao%20do%20Codigo%20de%20Liquidacao%20das%20Opcoes.pdf) · [Receita Federal – Bolsa de valores](https://www.gov.br/receitafederal/pt-br/assuntos/meu-imposto-de-renda/pagamento/renda-variavel/bolsa-de-valores-1) · [MQL5 Python docs](https://www.mql5.com/en/docs/python_metatrader5)

**Keywords:** B3, derivativos, opções, options, futuros, futures, mini-índice, WIN, mini-dólar, WDO, IND, DOL, DI1, taxa DI, cupom cambial, boi gordo, BGI, café arábica, ICF, milho, CCM, soja, SFI, opções sobre ações, calls, puts, strike, vencimento, exercício, americana, europeia, margem de garantia, alavancagem, day trade, CORE clearing, COE, certificado de operações estruturadas, MetaTrader5, ProfitChart, Nelogica, market data, machine learning, quant, volatilidade, Ibovespa, derivatives, Brazil, Brasil, hedge, opcoes.net.br, brapi
