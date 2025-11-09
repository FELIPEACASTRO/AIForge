# Forecasting Prompts (Prompts de Previsão)

## Description
**Forecasting Prompts** é uma técnica de engenharia de prompt que utiliza Large Language Models (LLMs) para realizar tarefas de **previsão de séries temporais (Time Series Forecasting - TSF)**. Em vez de depender exclusivamente de modelos estatísticos ou arquiteturas de aprendizado de máquina tradicionais, esta abordagem traduz os dados de séries temporais e o problema de previsão em um formato compreensível pelo LLM, geralmente texto estruturado ou sequências de tokens.

A inovação mais notável é o conceito de **Patch-Based Prompting** (Prompting Baseado em Patches), como exemplificado pelo framework **PatchInstruct** [1]. Esta técnica divide a série temporal em segmentos menores, chamados "patches", que encapsulam padrões temporais locais. Esses patches são então tokenizados e apresentados ao LLM junto com instruções em linguagem natural, permitindo que o modelo utilize suas capacidades de reconhecimento de padrões e modelagem de sequência para gerar previsões futuras [1]. O principal benefício é a capacidade de realizar previsões precisas e eficientes sem a necessidade de *fine-tuning* extensivo ou modificações arquitetônicas complexas, reduzindo drasticamente o *overhead* de inferência [1].

## Examples
```
1.  **Previsão de Vendas (Zero-Shot)**
    ```
    **Instrução:** "Você é um analista de dados. Preveja os próximos 7 dias de vendas com base nos dados históricos fornecidos. O resultado deve ser uma lista de 7 números inteiros.
    **Dados Históricos (Vendas Diárias):** [120, 135, 140, 155, 160, 145, 170, 185, 190, 205, 210, 225, 230, 245]"
    ```

2.  **Previsão de Tráfego (Patch-Based - Conceitual)**
    ```
    **Instrução:** "Analise os seguintes 'patches' de dados de tráfego (média de veículos por hora) da última semana. O patch 1 é a tendência de segunda-feira, o patch 2 é a de terça-feira, e assim por diante. Preveja o valor médio de tráfego para o pico da próxima segunda-feira (Patch 8).
    **Patch 1 (Seg):** [450, 510, 620]
    **Patch 2 (Ter):** [460, 525, 635]
    **Patch 3 (Qua):** [445, 505, 615]
    **Patch 4 (Qui):** [470, 530, 640]
    **Patch 5 (Sex):** [500, 600, 750]
    **Patch 6 (Sáb):** [300, 350, 400]
    **Patch 7 (Dom):** [250, 300, 320]"
    ```

3.  **Previsão de Preços de Ações (Com Contexto)**
    ```
    **Instrução:** "Com base nos preços de fechamento dos últimos 30 dias e considerando que a empresa anunciou um novo produto de sucesso ontem, preveja o preço de fechamento da ação para os próximos 5 dias. Forneça a previsão como uma lista de floats.
    **Preços de Fechamento:** [50.2, 51.1, 50.8, 52.5, 53.0, 54.1, 55.5, 56.0, 55.8, 57.2, ... (mais 20 valores)]"
    ```

4.  **Previsão de Demanda de Energia (Com Sazonalidade)**
    ```
    **Instrução:** "A série temporal representa o consumo de energia (em MWh) por hora nas últimas 48 horas. Identifique o padrão sazonal diário e preveja o consumo para as próximas 6 horas.
    **Série Temporal:** [45, 42, 40, 41, 48, 55, 65, 72, 70, 68, 65, 60, 55, 50, 48, 45, 42, 40, 41, 48, 55, 65, 72, 70, 68, 65, 60, 55, 50, 48, 45, 42, 40, 41, 48, 55, 65, 72, 70, 68, 65, 60, 55, 50, 48, 45, 42, 40]"
    ```

5.  **Previsão de Clima (Multivariada Simplificada)**
    ```
    **Instrução:** "Preveja a temperatura máxima (Tmax) e a precipitação (P) para os próximos 3 dias. Use os dados diários das últimas 10 observações.
    **Dados:**
    Dia 1: Tmax=25, P=0
    Dia 2: Tmax=26, P=0
    Dia 3: Tmax=24, P=5
    Dia 4: Tmax=22, P=15
    Dia 5: Tmax=23, P=10
    Dia 6: Tmax=25, P=0
    Dia 7: Tmax=27, P=0
    Dia 8: Tmax=28, P=0
    Dia 9: Tmax=26, P=2
    Dia 10: Tmax=25, P=5
    **Formato de Saída:** [Tmax Dia 11, P Dia 11], [Tmax Dia 12, P Dia 12], [Tmax Dia 13, P Dia 13]"
    ```

6.  **Previsão de Anomalias (Instrução de Detecção)**
    ```
    **Instrução:** "A série temporal representa a latência de um servidor (em ms). Preveja o próximo valor e, mais importante, determine se o valor previsto é uma anomalia (Sim/Não) com base no desvio padrão dos dados de entrada.
    **Latência:** [50, 52, 51, 53, 50, 54, 55, 52, 51, 50, 53, 52, 51, 50, 52, 51, 50, 53, 52, 51]"
    ```

7.  **Previsão com Decomposição (Conceitual)**
    ```
    **Instrução:** "A série temporal foi decomposta em Tendência, Sazonalidade e Resíduo. Preveja o próximo valor da Tendência e da Sazonalidade, e combine-os para a previsão final.
    **Tendência:** [100, 105, 110, 115, 120]
    **Sazonalidade:** [5, -2, -3, 5, -2]
    **Resíduo:** [0.1, -0.5, 0.2, 0.0, -0.1]"
    ```
```

## Best Practices
*   **Tokenização Eficiente (Patching):** Em vez de despejar a série temporal completa, utilize técnicas como o *Patch-Based Prompting* para tokenizar a série em segmentos significativos. Isso reduz o uso de tokens e permite que o LLM capture padrões temporais de curto prazo de forma mais eficaz [1] [2].
*   **Instrução Estruturada:** Defina claramente o papel do LLM ("Você é um analista de dados..."), o horizonte de previsão (quantos passos à frente) e o formato de saída desejado (lista, JSON, número único) [2].
*   **Normalização e Escala:** Embora os LLMs sejam robustos, a normalização dos dados de entrada (por exemplo, para um intervalo de 0 a 1 ou escores Z) pode melhorar a estabilidade e a precisão da previsão.
*   **Fornecer Contexto:** Inclua metadados relevantes (feriados, eventos de marketing, mudanças regulatórias) que possam influenciar a série temporal, pois os LLMs são excelentes em incorporar informações textuais [2].
*   **Decomposição Explícita:** Para séries temporais complexas, decomponha-as em componentes (tendência, sazonalidade, resíduo) e forneça esses componentes separadamente ao LLM, instruindo-o a prever cada parte e depois recombiná-las [1].

## Use Cases
*   **Finanças e Comércio:** Previsão de preços de ações, volumes de negociação, demanda de produtos em e-commerce e gerenciamento de inventário [2].
*   **Energia e Utilidades:** Previsão de demanda de eletricidade, consumo de gás e produção de energia renovável.
*   **Logística e Cadeia de Suprimentos:** Previsão de remessas, atrasos na entrega e necessidades de capacidade de armazenamento [2].
*   **Saúde:** Previsão de surtos de doenças, demanda hospitalar e consumo de medicamentos.
*   **Monitoramento de Sistemas:** Previsão de latência de servidores, uso de CPU e detecção de anomalias em logs de sistemas.

## Pitfalls
*   **Alucinação Numérica:** LLMs podem "alucinar" números ou sequências que parecem plausíveis, mas não são matematicamente válidas ou coerentes com os dados de entrada.
*   **Limitação de Contexto (Token Limit):** Séries temporais longas podem exceder o limite de tokens do LLM, exigindo técnicas de sumarização ou *patching* que podem levar à perda de informações cruciais [2].
*   **Ignorar Dependências Complexas:** LLMs podem ter dificuldade em modelar dependências inter-séries (em previsões multivariadas) ou relações de longo prazo sem instruções explícitas ou arquiteturas auxiliares [1].
*   **Sensibilidade ao Formato:** A precisão da previsão é altamente sensível à forma como os dados são formatados e apresentados no prompt. Um formato de dados inconsistente ou ambíguo pode levar a resultados errados.
*   **Viés de Treinamento:** O LLM pode introduzir vieses de seu treinamento geral (por exemplo, conhecimento de eventos mundiais) que podem não ser relevantes ou podem distorcer a previsão puramente baseada em dados.

## URL
[https://arxiv.org/html/2506.12953v1](https://arxiv.org/html/2506.12953v1)
