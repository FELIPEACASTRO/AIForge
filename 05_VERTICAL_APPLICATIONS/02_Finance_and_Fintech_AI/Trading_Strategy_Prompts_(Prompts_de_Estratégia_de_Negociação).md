# Trading Strategy Prompts (Prompts de Estratégia de Negociação)

## Description
**Prompts de Estratégia de Negociação** são instruções estruturadas e detalhadas fornecidas a Modelos de Linguagem Grande (LLMs) para auxiliar na criação, otimização, backtesting e análise de estratégias de trading nos mercados financeiros (ações, forex, criptomoedas, commodities). A técnica se baseia em engenharia de prompt avançada, como o **Role Prompting** e o **Chain-of-Thought**, para forçar a IA a operar como um analista quantitativo ou desenvolvedor de algoritmos. O objetivo principal não é prever o mercado, mas sim automatizar a geração de lógica de negociação, código de backtesting (em Python, Pine Script, MQL5, etc.), e a análise de risco e desempenho. A eficácia reside na capacidade de fornecer contexto financeiro específico, dados de mercado (ou a estrutura para eles) e restrições de risco claras, transformando a IA em uma ferramenta poderosa para a pesquisa e desenvolvimento de estratégias algorítmicas.

## Examples
```
**1. Geração de Código para Estratégia de Média Móvel**

```
**Papel:** Atue como um desenvolvedor de algoritmos de trading sênior.
**Tarefa:** Gere o código completo em Python (usando a biblioteca `pandas` e `backtesting.py`) para uma estratégia de cruzamento de Médias Móveis.
**Lógica:**
1. Média Móvel Rápida (SMA) de 20 períodos.
2. Média Móvel Lenta (SMA) de 50 períodos.
3. **Sinal de Compra:** Quando a SMA Rápida cruza acima da SMA Lenta.
4. **Sinal de Venda:** Quando a SMA Rápida cruza abaixo da SMA Lenta.
**Saída Esperada:** O código Python completo, incluindo a classe da estratégia e a função de backtesting.
```

**2. Otimização de Parâmetros e Análise de Risco**

```
**Papel:** Atue como um Estrategista de Risco Quantitativo.
**Contexto:** A estratégia é um "Breakout de Volatilidade" no Bitcoin (BTC/USD) em um gráfico de 4 horas.
**Parâmetros Atuais:** Janela do ATR (Average True Range) = 14; Multiplicador de Stop Loss = 2.5.
**Tarefa:** Sugira 3 conjuntos de parâmetros otimizados (ATR e Multiplicador) que visem maximizar o Fator de Lucro (Profit Factor) e, ao mesmo tempo, manter o Drawdown Máximo (Max Drawdown) abaixo de 15%.
**Saída Esperada:** Uma tabela comparativa com os 3 conjuntos de parâmetros, o Fator de Lucro estimado e o Drawdown Máximo.
```

**3. Criação de um Sistema de Gerenciamento de Risco**

```
**Papel:** Atue como um Gerente de Portfólio.
**Tarefa:** Desenvolva um sistema de gerenciamento de risco (Money Management) para uma conta de trading de $50.000.
**Regras:**
1. Risco máximo por negociação: 1% do capital total.
2. Risco máximo diário: 3% do capital total.
3. Calcule o tamanho da posição (em unidades) para uma negociação onde o Stop Loss está a 50 pips de distância.
**Saída Esperada:** O cálculo do tamanho da posição e um resumo das regras de risco em formato de lista.
```

**4. Análise Técnica Baseada em Indicadores**

```
**Papel:** Atue como um Analista Técnico de Mercado.
**Ativo:** Ações da Tesla (TSLA).
**Intervalo:** Diário.
**Indicadores:** Bandas de Bollinger (20, 2) e MACD (12, 26, 9).
**Tarefa:** Analise a situação atual da TSLA com base nos indicadores fornecidos.
**Racional:** Descreva o que cada indicador está sinalizando (ex: preço tocando a banda inferior, MACD cruzando a linha de sinal).
**Conclusão:** Forneça um resumo de 3 pontos sobre o viés de negociação (altista, baixista ou neutro) e o nível de suporte/resistência mais próximo.
```

**5. Backtesting de Estratégia de Reversão à Média**

```
**Papel:** Atue como um Engenheiro de Backtesting.
**Estratégia:** Reversão à Média (Mean Reversion) no S&P 500 (SPY).
**Lógica:** Comprar quando o preço de fechamento estiver 2 desvios-padrão abaixo da Média Móvel de 20 dias. Vender quando o preço retornar à Média Móvel.
**Tarefa:** Descreva os 5 principais desafios de backtesting para esta estratégia (ex: custo de transação, slippage, volatilidade).
**Saída Esperada:** Uma lista numerada dos desafios e uma sugestão de como mitigar cada um.
```

**6. Prompt para Análise de Sentimento de Notícias**

```
**Papel:** Atue como um Analista de Sentimento de Mercado.
**Entrada:** [Insira aqui o texto de uma notícia recente sobre o Banco Central Europeu (BCE)].
**Tarefa:** Analise o texto e classifique o sentimento geral (Altista, Baixista, Neutro) para o par EUR/USD.
**Saída Esperada:**
1. **Sentimento:** [Classificação]
2. **Justificativa:** [Explicação concisa baseada no texto]
3. **Implicação de Trading:** [Sugestão de ação de trading de curto prazo]
```

**7. Prompt para Criação de um Plano de Trading**

```
**Papel:** Atue como um Coach de Trading Pessoal.
**Tarefa:** Crie um plano de trading estruturado para um trader iniciante focado em swing trading de ações.
**Seções Obrigatórias:**
1. Definição de Metas (Realistas)
2. Regras de Entrada e Saída (Genéricas)
3. Regras de Gerenciamento de Risco (Stop Loss e Tamanho da Posição)
4. Rotina Diária de Análise
5. Regras de Psicologia de Trading (Ex: Não negociar após 3 perdas consecutivas)
**Saída Esperada:** O plano de trading completo em formato de lista ou tópicos.
```
```

## Best Practices
**1. Defina o Papel (Role Prompting):** Comece o prompt instruindo a IA a agir como um "Estrategista de Trading Quantitativo Sênior", "Analista de Mercado com 10 Anos de Experiência" ou "Desenvolvedor de Algoritmos de Alta Frequência". Isso força o modelo a usar um vocabulário e um raciocínio mais especializados.
**2. Estrutura e Formato:** Use tags XML (`<ENTRADA>`, `<SAIDA>`, `<RACIONAL>`) ou formatação clara para delimitar as seções do seu prompt e da resposta esperada. Isso melhora a precisão e a capacidade de processamento da IA.
**3. Seja Específico e Contextual:** Em vez de pedir uma "estratégia de negociação", peça uma "estratégia de reversão à média para o par EUR/USD em um gráfico de 1 hora, usando o indicador RSI (14) com níveis de sobrecompra/sobrevenda em 70/30".
**4. Cadeia de Pensamento (Chain-of-Thought):** Peça à IA para primeiro detalhar o **racional** por trás da estratégia, depois o **código** e, por fim, os **parâmetros de risco**. Isso garante que o modelo "pense" antes de codificar, reduzindo alucinações.
**5. Forneça Dados de Exemplo:** Se possível, inclua um pequeno trecho de dados históricos (em formato CSV ou JSON) para que a IA possa basear sua análise em um contexto real, mesmo que seja apenas para fins de demonstração da lógica.
**6. Foco no Processo, Não na Previsão:** Use prompts para otimizar o processo (backtesting, gerenciamento de risco, análise de indicadores) em vez de tentar obter previsões diretas do preço futuro, que são inerentemente não confiáveis.

## Use Cases
**1. Desenvolvimento de Estratégias Algorítmicas:** Geração de código funcional (Python, Pine Script) para novas estratégias de trading, como arbitragem, momentum ou reversão à média.
**2. Otimização de Parâmetros:** Identificação de parâmetros ideais (ex: períodos de Média Móvel, níveis de RSI) para maximizar o desempenho de uma estratégia existente em diferentes condições de mercado.
**3. Backtesting e Simulação:** Criação de estruturas de backtesting e análise de métricas de desempenho (Sharpe Ratio, Drawdown, Profit Factor) para validar a robustez da estratégia.
**4. Gerenciamento de Risco:** Desenvolvimento de regras de gerenciamento de capital e cálculo de tamanho de posição para limitar a exposição ao risco.
**5. Análise de Sentimento:** Processamento de grandes volumes de notícias financeiras ou dados de redes sociais para extrair um viés de sentimento (altista/baixista) que possa ser integrado à estratégia.
**6. Educação e Aprendizado:** Geração de explicações detalhadas sobre conceitos complexos de trading, como Teoria das Ondas de Elliott ou modelos de precificação de opções.
**7. Criação de Planos de Trading:** Estruturação de planos de trading pessoais e rotinas de análise de mercado.

## Pitfalls
**1. Alucinação de Dados:** A IA pode inventar dados históricos, resultados de backtesting ou indicadores técnicos. **Mitigação:** Use a IA apenas para gerar a lógica e o código; execute o backtesting em plataformas de trading reais com dados verificados.
**2. Overfitting (Ajuste Excessivo):** Criar prompts que resultam em estratégias excessivamente otimizadas para um conjunto de dados específico. **Mitigação:** Peça à IA para incluir uma seção de "Robustez da Estratégia" e sugerir testes fora da amostra (Out-of-Sample Testing).
**3. Falta de Contexto Financeiro:** Usar prompts genéricos sem definir o ativo, o intervalo de tempo (timeframe) e as condições de mercado. **Mitigação:** Sempre use o **Role Prompting** e forneça o máximo de contexto técnico e fundamentalista possível.
**4. Confundir Previsão com Análise:** Pedir à IA para "prever o preço de amanhã". **Mitigação:** Concentre-se em prompts que analisam a **probabilidade** de um movimento com base em condições predefinidas (ex: "Qual a probabilidade de alta se o RSI estiver abaixo de 30?").
**5. Ignorar o Risco:** Focar apenas no lucro e negligenciar o gerenciamento de risco no prompt. **Mitigação:** Torne o gerenciamento de risco (Stop Loss, Tamanho da Posição, Drawdown Máximo) um componente obrigatório da saída do prompt.

## URL
[https://roguequant.substack.com/p/prompt-engineering-for-traders-how](https://roguequant.substack.com/p/prompt-engineering-for-traders-how)
