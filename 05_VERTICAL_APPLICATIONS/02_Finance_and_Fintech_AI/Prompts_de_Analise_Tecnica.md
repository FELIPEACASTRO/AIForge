# Prompts de Análise Técnica

## Description
Prompts de Análise Técnica (Technical Analysis Prompts) são instruções estruturadas e detalhadas fornecidas a Grandes Modelos de Linguagem (LLMs) com o objetivo de simular a análise de dados de mercado, como preços e volumes, para identificar padrões, tendências e sinais de negociação. Essa técnica de engenharia de prompt é crucial no setor financeiro, especialmente para traders e analistas que buscam automatizar ou acelerar a interpretação de indicadores técnicos (como RSI, MACD, Médias Móveis) e padrões gráficos (como Cabeça e Ombros, Triângulos).

A eficácia desses prompts reside na capacidade de fornecer ao LLM o **contexto** necessário (dados históricos, indicadores específicos, horizonte temporal) e o **papel** (analista financeiro, trader quantitativo) para que ele possa aplicar o raciocínio analítico e estatístico inerente ao seu treinamento. O prompt deve ser claro, conciso e, idealmente, solicitar uma saída estruturada para facilitar a integração com sistemas de negociação ou relatórios.

## Examples
```
1.  **Análise de Indicadores Múltiplos:**
    ```
    Aja como um analista quantitativo. Analise o ativo [NOME DO ATIVO/TICKER] com base nos seguintes dados: [INSERIR DADOS DE PREÇO E VOLUME]. Calcule e interprete o RSI (14 períodos) e o MACD (12, 26, 9). O RSI está em [VALOR] e o MACD está [ACIMA/ABAIXO] da linha de sinal. Forneça uma conclusão sobre o momentum e a tendência, e sugira um sinal de negociação (Compra, Venda, Neutro).
    ```

2.  **Identificação de Padrão Gráfico:**
    ```
    Você é um especialista em padrões de velas japonesas. Dada a sequência de preços de fechamento dos últimos 30 dias para [NOME DO ATIVO], identifique se há um padrão de reversão ou continuação (ex: Martelo, Estrela da Manhã, Engolfo de Baixa). Se um padrão for encontrado, descreva-o e explique sua implicação de preço esperada.
    ```

3.  **Estratégia de Cruzamento de Médias Móveis:**
    ```
    Crie uma estratégia de negociação baseada no cruzamento das Médias Móveis Exponenciais (MME) de 9 e 21 períodos para o ativo [NOME DO ATIVO]. Defina as regras de entrada e saída: Comprar quando MME(9) cruzar acima de MME(21). Vender quando MME(9) cruzar abaixo de MME(21). Avalie a eficácia dessa estratégia nos últimos 6 meses (com base nos dados fornecidos) e sugira um Stop Loss percentual.
    ```

4.  **Análise de Volatilidade (Bandas de Bollinger):**
    ```
    Aja como um gestor de risco. Para o ativo [NOME DO ATIVO], as Bandas de Bollinger (20 períodos, 2 desvios padrão) estão [LARGAS/ESTREITAS]. O preço atual está [ACIMA/ABAIXO/DENTRO] da banda superior/inferior. Interprete este cenário em termos de volatilidade e probabilidade de um movimento de preço significativo. Qual seria uma entrada de negociação de alta probabilidade neste contexto?
    ```

5.  **Combinação de Análise Técnica e Sentimento:**
    ```
    Você é um analista de mercado. O ativo [NOME DO ATIVO] está mostrando um padrão de "Triângulo Ascendente" (padrão de continuação de alta). No entanto, a análise de sentimento nas redes sociais (dados fornecidos) indica um pessimismo crescente. Sintetize essas duas informações conflitantes e forneça uma recomendação de negociação ponderada, justificando qual fator (Técnico ou Sentimento) deve ter maior peso no momento.
    ```

6.  **Cálculo e Interpretação de Níveis de Fibonacci:**
    ```
    O ativo [NOME DO ATIVO] teve uma alta de [PREÇO MÍNIMO] para [PREÇO MÁXIMO]. Calcule os níveis de retração de Fibonacci de 38.2%, 50% e 61.8%. Se o preço estiver atualmente no nível de 50%, descreva a importância desse nível como suporte/resistência e qual a próxima zona de preço a ser observada.
    ```

7.  **Prompt de Refinamento (Chain-of-Thought):**
    ```
    Passo 1: Analise o gráfico de [NOME DO ATIVO] e identifique a tendência primária (curto, médio e longo prazo).
    Passo 2: Calcule o ADX (Average Directional Index) para medir a força dessa tendência.
    Passo 3: Com base na tendência e na força do ADX, gere um relatório de 100 palavras sobre a saúde do ativo.
    ```
```

## Best Practices
*   **Definir o Papel (Role-Playing):** Comece o prompt definindo o LLM como um "Analista Financeiro", "Trader Quantitativo" ou "Especialista em Risco". Isso direciona o tom e o foco da resposta.
*   **Fornecer Dados Estruturados:** Em vez de apenas descrever, forneça dados de preço e volume em um formato estruturado (tabela, CSV, ou lista de pontos-chave) para que o LLM possa processá-los com precisão.
*   **Especificar Indicadores e Parâmetros:** Mencione explicitamente quais indicadores técnicos devem ser usados (ex: MME de 200, RSI de 14, MACD com configurações padrão) e o horizonte temporal (diário, semanal, 4 horas).
*   **Solicitar Justificativa (Chain-of-Thought):** Peça ao LLM para detalhar o raciocínio por trás da conclusão (ex: "Explique por que o RSI em 75 indica sobrecompra antes de dar o sinal de venda"). Isso aumenta a transparência e a confiabilidade.
*   **Definir o Formato de Saída:** Peça a saída em um formato específico (ex: "Forneça a resposta em uma tabela com colunas: Indicador, Valor, Interpretação, Ação Sugerida").

## Use Cases
*   **Geração de Sinais de Negociação:** Criação de alertas automatizados de Compra/Venda com base na interpretação de múltiplos indicadores.
*   **Backtesting de Estratégias:** Simulação rápida da performance de uma estratégia de negociação em dados históricos sem a necessidade de codificação complexa.
*   **Educação e Treinamento:** Explicação de conceitos complexos de análise técnica (ex: Ondas de Elliott, Teoria de Dow) de forma simplificada e com exemplos práticos.
*   **Relatórios de Mercado:** Geração de resumos diários ou semanais sobre a situação técnica de um portfólio de ativos.
*   **Análise de Risco:** Identificação de níveis críticos de suporte e resistência (usando Pivots ou Fibonacci) para definir ordens de Stop Loss e Take Profit.

## Pitfalls
*   **Alucinação de Dados:** O LLM pode "inventar" dados de preço ou valores de indicadores se não forem fornecidos explicitamente. **Sempre forneça os dados de entrada.**
*   **Generalização Excessiva:** Pedir uma análise técnica sem especificar o ativo ou o horizonte temporal resultará em uma resposta genérica e inútil.
*   **Confundir Análise Técnica com Fundamentalista:** Misturar os dois tipos de análise no mesmo prompt sem um papel claro pode levar a conclusões confusas. Mantenha o foco estritamente nos dados de preço/volume, a menos que a combinação seja intencional e bem definida.
*   **Dependência Cega:** Tratar a saída do LLM como um sinal de negociação final. A análise de IA deve ser uma ferramenta de suporte, não um substituto para a tomada de decisão humana e a gestão de risco.
*   **Ausência de Contexto de Volatilidade:** Não considerar o contexto de mercado (alta ou baixa volatilidade) ao interpretar indicadores pode levar a sinais falsos.

## URL
[https://blog.galaxy.ai/chatgpt-prompts-for-trading](https://blog.galaxy.ai/chatgpt-prompts-for-trading)
