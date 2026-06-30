# B3 (Brazil) Options Strategies — Estratégias de Opções na B3

> A practical, Brazil-specific reference for trading options on **B3** (Brasil, Bolsa, Balcão) — instrument mechanics (ticker codes, American vs European style, expirations), the local strategy vocabulary (*financiamento*, *travas*, *borboleta*, *condor*, *put protetora*, *rolagem*), the brutal liquidity reality, **B3 clearing margin** (*margem de garantia*), and **Brazilian taxation** (*tributação*) of options. Education/research only — **not investment advice** (*não é recomendação de investimento*).

---

## 1. Why B3 options are different (read this first)

Brazilian options share global payoff math (covered elsewhere in this folder) but trade in a market with very specific structure and constraints:

- **Hyper-concentrated liquidity.** The Brazilian options market is one of the most concentrated in the world. By trading volume the leaders are **PETR4** (Petrobras), **BBAS3** (Banco do Brasil), **VALE3** (Vale), **ITUB4** (Itaú) and **BOVA11** (the iShares Ibovespa ETF). PETR4 options are routinely cited as among the **most liquid single-name options on the planet** ([opcoes.net.br liquidity study](https://opcoes.net.br/estudos/liquidez/opcoes); [Invius — liquidez de opções](https://www.invius.com.br/3606-liquidez-opcoes/)). In the first four months of B3's weekly-options program (2024–2025) the concentration was extreme: PETR4 ~35%, BOVA11 ~28%, VALE3 ~12% of weekly contracts traded ([B3 — novos tickers de opções semanais](https://www.b3.com.br/pt_br/noticias/novos-tickers-de-opcoes-semanais.htm)). **Outside the top ~10 names, bid/ask is wide, OTM strikes are dead, and "model" multi-leg structures may be unfillable.**
- **Equity options are mostly American-style; index options are European-style.** B3 stock options can be American (*americana* — exercisable any day) or European (*europeia* — only at expiry), set per series; index (Ibovespa) options are **European with cash settlement** ([B3 Edu — Opções sobre Ações](https://edu.b3.com.br/w/opcoes-acoes); [B3 — Opções sobre Ibovespa](https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/renda-variavel/opcoes-sobre-ibovespa.htm)). **American style = early-assignment risk**, especially around dividends (*proventos*) for ITM calls.
- **Puts are far less liquid than calls.** Brazilian retail culture historically traded calls (covered-call *financiamento*); listed put liquidity is thinner, which matters for *put protetora* and bear structures.
- **Centralized CCP clearing.** All margin/risk runs through the **B3 clearing house** using the **CORE** model (Close-Out Risk Evaluation) ([B3 CORE](https://www.b3.com.br/pt_br/produtos-e-servicos/compensacao-e-liquidacao/clearing/administracao-de-riscos/modelo-de-risco/core/)).

---

## 2. Contract mechanics & ticker codes (nomenclatura)

| Item | Stock options (*opções sobre ações*) | Index options (*opções sobre Ibovespa*) |
|---|---|---|
| Underlying | A share/unit/ETF/BDR (e.g. PETR4, VALE3, BOVA11) | Ibovespa index (cash) |
| Exercise style | **American or European** (per series) | **European only** |
| Settlement | Physical delivery of shares on exercise | **Cash** (financial), auto-settled at expiry |
| Standard lot | Tied to underlying lot (commonly **100 shares** per contract) | Quoted in **index points**; each point = **R$ 0.01** (contract reduced 100× from the former R$ 1/point) |
| Tick | R$ 0.01 | R$ 0.01 |
| Monthly expiry | **3rd Friday** of the month (postponed to next business day on holidays) | **Wednesday closest to the 15th** (next business day if no session) — *not* the 3rd Friday |
| Weekly expiry | **Every Friday except the 3rd Friday** (stock/ETF weeklies launched Jan/2024) | Weekly Ibovespa options expire on **Wednesdays** (since Feb/2025; later expanded to all business days) |

Sources: [B3 Edu — Opções sobre Ações](https://edu.b3.com.br/w/opcoes-acoes), [B3 — Opções Semanais sobre Ações](https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/renda-variavel/opcoes-semanais-sobre-acoes.htm), [B3 — Opções sobre Ibovespa](https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/renda-variavel/opcoes-sobre-ibovespa.htm), [B3 vencimentos calendar](https://www.b3.com.br/pt_br/solucoes/plataformas/puma-trading-system/para-participantes-e-traders/calendario-de-negociacao/vencimentos/calendario-de-vencimentos-de-opcoes-sobre-acoes-e-indices/).

### 2.1 Decoding the option ticker (the series letter)

A B3 option ticker = **underlying root + month/type letter + strike code**, e.g. `PETRX339`.

| Letter | Call (month) | Letter | Put (month) |
|---|---|---|---|
| **A** | January call | **M** | January put |
| **B** | February call | **N** | February put |
| **C** | March call | **O** | March put |
| **D** | April call | **P** | April put |
| **E** | May call | **Q** | May put |
| **F** | June call | **R** | June put |
| **G** | July call | **S** | July put |
| **H** | August call | **T** | August put |
| **I** | September call | **U** | September put |
| **J** | October call | **V** | October put |
| **K** | November call | **W** | November put |
| **L** | December call | **X** | December put |

So **A–L = calls (Jan→Dec)**, **M–X = puts (Jan→Dec)**. The trailing number encodes the **strike** (assigned per series; it is *not* literally the price, so two strikes get two distinct numbers). Always confirm the exact strike on the chain, not from the number alone ([B3 Edu — Opções](https://edu.b3.com.br/w/opcoes); [brapi.dev — guia de opções](https://brapi.dev/blog/opcoes-b3-guia-completo-iniciantes-calls-puts)). Weekly series use distinct roots/tickers ([B3 weekly tickers](https://www.b3.com.br/pt_br/noticias/novos-tickers-de-opcoes-semanais.htm)).

---

## 3. Core strategies in Brazilian terms

The global payoff theory is detailed in the sibling pages (Directional & Spreads; Volatility/Income/Neutral; Position Management). Here we map them to the **local name, structure, and the BR-specific risk** you must respect.

| Estratégia (PT) | Global name | Estrutura (mesma série/vencimento salvo nota) | Quando usar | Risco principal (BR) |
|---|---|---|---|---|
| **Financiamento / Lançamento coberto** | Covered call / buy-write | Long 100 ações + sell 1 call (geralmente OTM) | Renda sobre ação que você já tem; visão neutra-a-leve-alta | American calls → **exercício antecipado** perto de dividendos; upside capado |
| **Trava de alta com call** (*bull call spread*) | Bull call spread (débito) | Buy call strike menor + sell call strike maior | Alta moderada, custo reduzido | Perda máx. = débito; precisa subir |
| **Trava de alta com put** (*bull put spread*) | Bull put spread (crédito) | Sell put strike maior + buy put strike menor | Alta/lateral, vender prêmio com risco definido | Put liquidity fraca; risco = largura − crédito |
| **Trava de baixa com put** (*bear put spread*) | Bear put spread (débito) | Buy put strike maior + sell put strike menor | Queda moderada | Perda máx. = débito |
| **Trava de baixa com call** (*bear call spread*) | Bear call spread (crédito) | Sell call strike menor + buy call strike maior | Queda/lateral, vender prêmio com teto | Short call → assignment; risco = largura − crédito |
| **Borboleta** (*butterfly*) | Long butterfly | Trava de alta + trava de baixa (1×2×1) | Aposta em preço **parado** num strike; baixo custo | Liquidez de 3 strikes; difícil montar/fechar fora do top |
| **Condor / Iron Condor** | Condor / iron condor | 4 strikes (2 compradas, 2 vendidas) | Lateralização em faixa ampla | 4 pernas = 4× spread/custos; execução ruim em BR |
| **Mesa / boleta de combinação** | Multi-leg combo order | Várias pernas em ordem única (estruturada) | Reduzir risco de execução perna-a-perna | Nem toda corretora suporta todas as combinações |
| **Venda coberta** (*covered write*) | Covered short | Vender call **com a ação em carteira** | Renda; risco limitado pela ação | Custo de oportunidade; chamada antecipada |
| **Venda descoberta** (*naked/uncovered*) | Naked short option | Vender call/put **sem o ativo** | Vender prêmio agressivo | **Risco ilimitado (call) / grande (put)** + **margem alta** |
| **Put protetora** (*seguro de carteira*) | Protective put | Long ação + buy put OTM | Proteger ganho/posição contra queda | Prêmio = custo do "seguro"; **put illiquidity** |
| **Rolagem** | Roll | Fechar a perna atual e abrir em novo strike/vencimento | Estender prazo, ajustar perdedora, evitar exercício | Pode "travar" prejuízo; custos × 2; *não* transforma trade ruim em bom |

Sources: [InfoMoney — 10 estratégias com opções](https://www.infomoney.com.br/mercados/perca-medo-dos-derivativos-10-estrategias-com-opcoes-para-ganhar-na-alta-baixa-ou-lateralizacao/), [Gorila — travas e borboletas](https://gorila.com.br/blog/estrategias-com-opcoes), [Nelogica — operações estruturadas](https://blog.nelogica.com.br/o-que-sao-operacoes-estruturadas-com-opcoes/), [Clear — eBooks de Renda Variável](https://www.clear.com.br/site/ebooks-renda-variavel).

### 3.1 Payoff / breakeven cheatsheet (per 1 contract = 100 shares)

| Estrutura | Lucro máx. | Perda máx. | Breakeven (ponto de equilíbrio) |
|---|---|---|---|
| Financiamento (S0, call K, prêmio c) | (K − S0) + c | S0 − c (até zero) | S0 − c |
| Trava de alta com call (débito D, largura L) | L − D | D | K_compra + D |
| Trava de alta com put (crédito C, largura L) | C | L − C | K_venda − C |
| Trava de baixa com put (débito D, largura L) | L − D | D | K_compra − D |
| Trava de baixa com call (crédito C, largura L) | C | L − C | K_venda + C |
| Borboleta (débito D, asas em ±) | L − D (no strike central) | D | central ± (L − D) |
| Put protetora (ação S0, put K, prêmio p) | ilimitado (− p) | S0 − K + p | S0 + p |

L = largura entre strikes (× 100). Sinais simplificados; confirme na boleta da corretora.

---

## 4. "Financiamento": the dominant Brazilian trade

The single most popular structured options trade in Brazil is the **financiamento** (covered call / *lançamento coberto*): **buy 100 shares + sell 1 call**. It is marketed as a way to earn a defined return ("*renda sintética*" / synthetic fixed-income-like yield) on a stock position while accepting a capped upside ([Itaú — financiamento coberto](https://www.itaucorretora.com.br/nossosservicos/financiamento-coberto-de-opcoes.aspx); [BTG — lançamento coberto](https://content.btgpactual.com/blog/renda-variavel/lancamento-coberto-de-opcoes); [Ágora — Financiamento (PDF)](https://www.agorainvest.com.br/uploads/centro_informacoes/operacoes_estruturadas/artigos/Financiamento.pdf)).

- **Taxa de financiamento** = the implied return from `(strike + premium − stock price)`; a deep-ITM call sale ("*financiamento dentro do dinheiro*") locks in a more bond-like, lower-variance return.
- **Honest risks:** (1) upside is **capped** at the strike; (2) American calls can be **assigned early** before ex-dividend (*data com/ex proventos*) — the call buyer captures the dividend by exercising, taking your shares; (3) it is **not** free yield — it is short a call's convexity, so a crash still hurts the stock leg minus the small premium.

---

## 5. Mesa de opções, vencimento, exercício, margem

- **Mesa de opções / boleta de combinações** — Brazilian platforms let you send **multi-leg structured orders** ("operações estruturadas") as a single boleta (ticket) so legs fill together, reducing leg-risk. Support varies by broker/platform ([Nelogica — operações estruturadas](https://blog.nelogica.com.br/o-que-sao-operacoes-estruturadas-com-opcoes/)).
- **Vencimento (expiry)** — for **stock/ETF options**, monthly = **3rd Friday** and weeklies (since Jan/2024) = **every Friday except the 3rd**. **Ibovespa index options follow a different calendar**: monthly = the **Wednesday closest to the 15th**, with weeklies on Wednesdays (since 2025, later expanded to all business days). ITM options are **auto-exercised** (*exercício automático*) by B3 per series rules — do not assume an OTM-looking option is safe near close ([B3 Edu](https://edu.b3.com.br/w/opcoes-acoes); [B3 — Opções sobre Ibovespa](https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/renda-variavel/opcoes-sobre-ibovespa.htm)).
- **Exercício (exercise/assignment)** — American equity options: assignment can hit any time, randomly allocated to short holders; index options settle in **cash** at expiry.
- **Margem de garantia (clearing margin)** — sellers of uncovered options must post collateral computed by B3's **CORE** model, a portfolio-level close-out risk measure. Eligible collateral includes Tesouro/federal bonds, eligible shares, BDRs/ADRs, and fund quotes. Use B3's **Margin Simulator** to estimate calls ([B3 — Garantias](https://www.b3.com.br/pt_br/produtos-e-servicos/compensacao-e-liquidacao/clearing/administracao-de-riscos/garantias/garantias-aceitas/); [B3 — CORE](https://www.b3.com.br/pt_br/produtos-e-servicos/compensacao-e-liquidacao/clearing/administracao-de-riscos/modelo-de-risco/core/); [B3 — Margin Simulator](https://www.b3.com.br/pt_br/solucoes/plataformas/gestao-de-risco/risk-services/margin-simulator/)).

> **Venda descoberta warning (*venda a seco*):** selling naked calls = theoretically unlimited loss; naked puts = loss down to (strike − premium) × 100. Margin can be called intraday; a gap (e.g. a Petrobras headline) can blow past your collateral. Defined-risk *travas* exist precisely to cap this.

---

## 6. Macro / portfolio-level use in a Brazilian context

- **Seguro de carteira (portfolio hedge).** Buy puts on **BOVA11** or **IBOV options** to hedge a long Brazilian equity book — index options are European/cash-settled and avoid single-name assignment, but BR index put liquidity and cost must be checked vs. simply reducing exposure.
- **Macro/tail events that move BR vol:** Copom rate decisions (Selic), fiscal/political headlines, USD/BRL (câmbio) shocks, commodity moves (oil → PETR4, iron ore → VALE3), and US Fed/global risk-off. These can spike implied vol and crush short-premium structures.
- **Event clustering:** earnings (*balanços*) and ex-dividend dates drive single-name vol and **early-assignment** decisions on American calls — schedule trades around them.
- **Concentration is your macro risk:** because liquidity is so concentrated in PETR4/VALE3/financials, a "diversified" BR options book is often a **disguised oil + iron ore + rates bet**. Size accordingly.

---

## 7. Taxation 🇧🇷 (tributação de opções) — current rules

Confirm with a contador; rules change and this is not tax advice.

| Item | Rule | Note |
|---|---|---|
| **Swing trade** (operação comum) | **15%** sobre o lucro líquido | Apuração mensal |
| **Day trade** | **20%** sobre o lucro líquido | Mesma série comprada e vendida no dia |
| **IRRF "dedo-duro"** | **0,005%** retido na fonte sobre o **valor de venda** em operação comum/swing; **1%** sobre o **ganho** em **day trade** | Sinaliza a operação à Receita; é antecipação, compensável no DARF |
| **Isenção de R$ 20 mil/mês** | **NÃO se aplica a opções** | A isenção é só p/ venda de **ações** no à vista; opções são sempre tributadas |
| **Pagamento** | Via **DARF** (código **6015** de renda variável) até o **último dia útil do mês seguinte** | Você mesmo apura e paga |
| **Compensação de prejuízo** | Swing ↔ swing (qualquer RV); **day trade só com day trade** | Carrega para meses seguintes |

Sources: [XP — day trade no IR](https://conteudos.xpi.com.br/aprenda-a-investir/relatorios/day-trade-no-imposto-de-renda/), [investimentos.com.br — declarar opções](https://investimentos.com.br/artigos/como-declarar-opcoes-no-imposto-de-renda/), [Nubank — DARF day trade](https://blog.nubank.com.br/como-emitir-darf-day-trade/), [B3 Bora Investir — declarar day trade](https://borainvestir.b3.com.br/tipos-de-investimentos/renda-variavel/day-trade/como-declarar-day-trade-no-imposto-de-renda-confira-com-o-bora/), [Receita Federal — Isenções (renda variável)](https://www.gov.br/receitafederal/pt-br/assuntos/meu-imposto-de-renda/pagamento/renda-variavel/bolsa-de-valores-1/isencoes).

> Key trap: many beginners assume the R$ 20k/month equity exemption covers option profits. **It does not** — option gains (and any day trade) are taxable from the first real.
>
> Note (2025–2026): a proposed reform (MP 1.303) that would have unified equity/derivatives gains at a flat **17.5%** was **rejected by Congress and lapsed**, so the **15% swing / 20% day-trade** rates above remain in force for 2026.

---

## 8. Tools, data & brokers

| Tool / Data | What it does | URL |
|---|---|---|
| **opcoes.net.br** | Free/freemium options chain, quotes, **liquidity rankings**, studies | https://opcoes.net.br/ |
| **OpLab** | Options analysis/execution/simulation, chains, payoff graphs, **API** | https://oplab.com.br/ |
| **B3 — posições em aberto** | Official open-interest data | https://www.b3.com.br/pt_br/market-data-e-indices/servicos-de-dados/market-data/consultas/mercado-a-vista/opcoes/posicoes-em-aberto/ |
| **B3 Margin Simulator / CORE** | Simulate margin (*margem*) on option positions | https://www.b3.com.br/pt_br/solucoes/plataformas/gestao-de-risco/risk-services/margin-simulator/ |
| **Profit / Nelogica** | Pro charting + structured-order desks | https://www.nelogica.com.br/ |
| **MetaTrader 5** | Order routing + options via supporting brokers | (broker-provided) |
| **TradeMap** | Market data app (BR retail) | https://www.trademap.com.br/ |

**Brokers (corretoras) with options access:** XP, Clear, Rico, BTG Pactual, Inter, among others — compare option fees, structured-order support, and assignment handling. (Clear publishes free [eBooks de Renda Variável](https://www.clear.com.br/site/ebooks-renda-variavel) covering options.)

---

## 9. Practical checklist before any B3 option trade

1. **Is it liquid?** If it's not in the top ~10 names (PETR4/VALE3/financials/BOVA11), check the actual bid/ask spread and open interest first.
2. **American or European?** Determines early-assignment risk — critical for short calls near dividends.
3. **Which expiry?** Stock monthly (3rd Fri) vs stock weekly (other Fridays) vs **Ibovespa (Wednesdays)** — match to your thesis horizon and the correct calendar.
4. **Margin you'll owe** if selling premium — run the **CORE/Margin Simulator**; assume intraday calls on gaps.
5. **Tax + DARF** — log the trade; remember the R$20k exemption does **not** apply; set a month-end DARF reminder.
6. **Exit/rolagem plan** before entry — define profit target, max loss, and assignment handling. Rolling does not fix a broken thesis.

---

## 10. Honest limitations & disclaimer

- BR option liquidity collapses fast outside the leaders; **textbook 3–4 leg structures (condors, borboletas) are often impractical to fill or close** at fair prices.
- Put liquidity is thinner than call liquidity → *put protetora* and bear strategies can be expensive/illiquid.
- American-style equity options carry **real early-assignment risk**, especially around *proventos*.
- Costs (corretagem, emolumentos, slippage) and the 15%/20% tax materially reduce edge — model EV **after** costs.

> **This page is research/education, not investment advice (não constitui recomendação de investimento).** Verify every fact against current **B3** and **CVM/Receita Federal** publications before trading.

---

**Keywords:** B3 options, opções B3, opções sobre ações, opções sobre Ibovespa, financiamento, lançamento coberto, covered call, trava de alta, trava de baixa, bull/bear spread, borboleta, butterfly, condor, iron condor, put protetora, protective put, seguro de carteira, venda coberta, venda descoberta, naked option, rolagem, roll, mesa de opções, boleta, vencimento, exercício, exercício antecipado, margem de garantia, CORE, clearing B3, PETR4, VALE3, BOVA11, IBOV, opções semanais, tributação de opções, DARF, day trade, swing trade, IRRF, dedo-duro, opcoes.net.br, OpLab, liquidez de opções.
