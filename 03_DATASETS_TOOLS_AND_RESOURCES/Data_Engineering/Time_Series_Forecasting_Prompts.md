# Time Series Forecasting Prompts

## Description
A técnica de *Time Series Forecasting Prompts* refere-se à engenharia de instruções estruturadas para Large Language Models (LLMs) com o objetivo de realizar tarefas complexas de análise e previsão de séries temporais. Ao invés de depender exclusivamente de modelos estatísticos ou de aprendizado de máquina tradicionais (como ARIMA, Prophet ou redes neurais especializadas), esta abordagem utiliza a capacidade de raciocínio e compreensão de contexto dos LLMs para identificar padrões, tendências, sazonalidade e anomalias nos dados de séries temporais. O sucesso reside na formatação cuidadosa dos dados de entrada (muitas vezes como sequências de texto ou "patches") e na definição clara da tarefa, das restrições e do formato de saída desejado.

## Examples
```
### 1. Previsão Zero-Shot com Contexto

**Objetivo:** Obter uma previsão de linha de base rápida para vendas trimestrais.

```
## System
Você é um especialista em análise de séries temporais focado no setor de varejo.
Sua tarefa é identificar padrões, tendências e sazonalidade para prever com precisão.

## User
Analise esta série temporal de vendas trimestrais: [120, 150, 130, 180, 140, 170, 150, 200, 160, 190, 170, 220]
- **Conjunto de dados:** Vendas trimestrais de uma loja de eletrônicos.
- **Frequência:** Trimestral
- **Features:** Apenas o valor das vendas.
- **Horizonte:** 4 trimestres à frente.

## Task
1. Preveja os próximos 4 trimestres.
2. Aponte os principais padrões sazonais ou de tendência observados.

## Constraints
- **Saída:** Lista em Markdown com as previsões (0 casas decimais).
- Adicione uma explicação de até 40 palavras sobre os fatores direcionadores.

## Evaluation Hook
Termine com: "Confiança: X/10. Premissas: [...]".
```

### 2. Patch-Based Prompting (PatchInstruct)

**Objetivo:** Prever a temperatura horária usando segmentos de dados recentes.

```
## System
You are a time-series forecasting expert in meteorology and sequential modeling.
Input: overlapping patches of size 3, reverse chronological (most recent first).

## User
Patches:
- Patch 1: [25.5, 25.4, 25.6]
- Patch 2: [25.8, 25.5, 25.4]
- Patch 3: [26.1, 25.8, 25.5]
...
- Patch N: [22.3, 22.1, 22.0]

## Task
1. Forecast next 3 values.
2. In ≤40 words, explain the recent trend.

## Constraints
- Output: Markdown list, 2 decimals.
- Ensure predictions align with the observed trend.

## Example
- Input: [20.0, 20.1, 20.2] → Output: [20.3, 20.4, 20.5].

## Evaluation Hook
Add: "Confidence: X/10. Assumptions: [...]".
```

### 3. Teste de Estacionariedade e Transformação

**Objetivo:** Verificar se uma série de preços de ações é estacionária e obter código para transformá-la se não for.

```
## System
Você é um analista de séries temporais quantitativo.

## User
- **Conjunto de dados:** Preço diário de fechamento da ação XYZ por 5 anos.
- **Frequência:** Diária
- **Tendência suspeita:** Não linear com volatilidade variável.
- **Contexto de negócio:** Análise de risco financeiro.

## Task
1. Explique como testar a estacionariedade usando:
   - Augmented Dickey-Fuller (ADF)
   - KPSS
   - Inspeção visual de gráficos
2. Se não for estacionária, sugira as transformações apropriadas (diferenciação, log, etc.).
3. Forneça código Python (usando `statsmodels` e `pandas`) para realizar os testes e as transformações.

## Constraints
- Mantenha a explicação com no máximo 120 palavras.
- O código deve estar pronto para copiar e colar.

## Evaluation Hook
Termine com: "Confiança: X/10. Premissas: [...]".
```

### 4. Análise de Autocorrelação (ACF/PACF)

**Objetivo:** Identificar lags significativos em dados de tráfego de rede para engenharia de features.

```
## System
Você é um especialista em séries temporais para monitoramento de redes.

## User
- **Conjunto de dados:** Volume de tráfego de rede (em GB) medido a cada 5 minutos.
- **Tamanho:** 2016 observações (1 semana).
- **Frequência:** 5 minutos.
- **Amostra bruta:** [1.2, 1.3, 1.2, 1.5, ...]

## Task
1. Forneça código Python para gerar gráficos ACF e PACF.
2. Explique como interpretar os gráficos para identificar:
   - Lags de Autoregressão (AR)
   - Componentes de Média Móvel (MA)
   - Padrões sazonais
3. Recomende features de lag com base nos lags significativos.
4. Mostre o código Python para criar essas features, tratando valores ausentes.

## Constraints
- Saída: Explicação de até 150 palavras + snippets de Python.
- Use `statsmodels` e `pandas`.

## Evaluation Hook
Termine com: "Confiança: X/10. Lags principais sinalizados: [listar lags]".
```

### 5. Detecção de Anomalias

**Objetivo:** Identificar leituras anômalas em sensores de uma máquina industrial.

```
## System
Você é um especialista em detecção de anomalias em dados de séries temporais de sensores IoT.

## User
- **Série temporal:** [20.1, 20.2, 20.0, 19.9, 20.1, 35.5, 20.3, ...]
- **Contexto:** Leitura de temperatura de um motor que opera continuamente.
- **Limites esperados:** 15°C a 30°C.

## Task
1. Identifique quaisquer pontos de dados que pareçam ser anomalias.
2. Para cada anomalia, forneça:
   - O valor anômalo.
   - O índice (ou timestamp) da anomalia.
   - Uma breve explicação do porquê é considerada uma anomalia (e.g., "pico repentino", "fora dos limites esperados").
3. Sugira uma estratégia para tratar/remover as anomalias antes da modelagem.

## Constraints
- Formate a saída como uma tabela Markdown.

## Evaluation Hook
Termine com: "Confiança na detecção: X/10. Premissas: [e.g., 'Anomalias são pontos únicos e não mudanças de regime']".
```
```

## Best Practices
1. **Patch-Based Prompting (PatchInstruct):** Dividir a série temporal em "patches" (segmentos) sobrepostos e alimentá-los ao LLM. Isso reduz o uso de tokens, preserva a interpretabilidade e permite que o modelo detecte padrões temporais de curto prazo de forma eficiente.
2. **Zero-Shot com Contexto:** Fornecer ao LLM uma descrição clara do conjunto de dados (domínio, frequência, horizonte de previsão) para que ele possa estabelecer uma linha de base de previsão sem treinamento adicional.
3. **Neighbor-Augmented Prompting:** Incluir séries temporais "vizinhas" ou correlacionadas no prompt para ajudar o LLM a identificar estruturas e padrões compartilhados, refinando a previsão para a série alvo.
4. **Estrutura de Prompt Detalhada:** Utilizar seções claras como `## System`, `## User`, `## Task`, `## Constraints` e `## Evaluation Hook` para guiar o modelo de forma precisa.
5. **Inclusão de Código:** Solicitar ao LLM que gere código Python (por exemplo, com `statsmodels` e `pandas`) para tarefas como teste de estacionariedade (ADF/KPSS), análise de autocorrelação (ACF/PACF) e decomposição sazonal (STL).

## Use Cases
1. **Previsão de Linha de Base Rápida:** Geração de previsões iniciais (zero-shot) para estabelecer um ponto de comparação.
2. **Análise Exploratória de Dados (EDA):** Identificação de estacionariedade, tendências, sazonalidade e anomalias.
3. **Engenharia de Features:** Geração de código para criar *features* de *lag*, janelas móveis e componentes cíclicos.
4. **Decomposição de Séries:** Separação da série em componentes de tendência, sazonalidade e resíduo para *insights* de negócios.
5. **Previsão em Domínios Específicos:** Aplicação em meteorologia, tráfego, vendas, finanças e outros domínios com dados sequenciais.

## Pitfalls
1. **Limitação de Contexto (Token Limit):** Tentar alimentar séries temporais muito longas diretamente no prompt, excedendo o limite de tokens e resultando em truncamento ou perda de informação.
2. **Alucinações e Falta de Rigor Estatístico:** LLMs podem gerar previsões plausíveis, mas estatisticamente incorretas ou sem aderência a testes de hipóteses formais (como estacionariedade).
3. **Formato de Dados Inadequado:** Não formatar os dados de forma estruturada (e.g., como *patches* ou listas claras), forçando o LLM a processar dados brutos de forma ineficiente.
4. **Ignorar a Estrutura Temporal:** Tratar a série temporal como um conjunto de dados comum, sem instruir o LLM sobre a dependência temporal, a frequência e o horizonte de previsão.
5. **Confiança Excessiva:** Confiar cegamente na previsão do LLM sem uma validação rigorosa com métricas de erro (MAE, RMSE) e testes de *backtesting*.

## URL
[https://towardsdatascience.com/prompt-engineering-for-time-series-analysis-with-large-language-models/](https://towardsdatascience.com/prompt-engineering-for-time-series-analysis-with-large-language-models/)
