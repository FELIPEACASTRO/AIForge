# Churn Analysis Prompts

## Description
**Análise de Churn (Churn Analysis)** é uma técnica de engenharia de prompt focada em utilizar Modelos de Linguagem Grande (LLMs) para processar dados de clientes, identificar padrões de comportamento que levam ao cancelamento (churn) e gerar *insights* acionáveis para retenção. Em vez de apenas prever o churn, esses prompts são projetados para atuar como um **analista de dados virtual**, realizando tarefas como segmentação de clientes, análise de causa raiz, criação de modelos preditivos (conceitualmente) e desenvolvimento de estratégias de intervenção.

A eficácia dos *Churn Analysis Prompts* reside na sua capacidade de estruturar a entrada de dados (muitas vezes via *copy-paste* de amostras de dados ou descrição de *features* de um dataset) e exigir uma saída analítica e estratégica, transformando dados brutos em inteligência de negócios. Eles são cruciais para empresas de SaaS, telecomunicações e serviços por assinatura, onde a retenção de clientes é diretamente ligada à saúde financeira.

**Churn Analysis (English)** is a prompt engineering technique focused on leveraging Large Language Models (LLMs) to process customer data, identify behavioral patterns that lead to cancellation (churn), and generate actionable insights for retention. Instead of merely predicting churn, these prompts are designed to act as a **virtual data analyst**, performing tasks such as customer segmentation, root cause analysis, conceptual predictive model creation, and intervention strategy development.

The effectiveness of *Churn Analysis Prompts* lies in their ability to structure data input (often via copy-pasting data samples or describing dataset features) and demand an analytical and strategic output, transforming raw data into business intelligence. They are crucial for SaaS, telecommunications, and subscription service companies, where customer retention is directly linked to financial health.

## Examples
```
**Exemplos de Prompts (Prompt Examples)**:

1.  **Análise de Cohort e Fatores de Risco (Cohort Analysis and Risk Factors)**
    ```
    **Papel:** Cientista de Dados Sênior.
    **Tarefa:** Analise o churn dos últimos 6 meses por cohort de aquisição (mensal).
    **Dados:** [Insira aqui um trecho de dados de churn em formato CSV ou Markdown, ou descreva as colunas: 'ID_Cliente', 'Data_Aquisicao', 'Meses_Ativo', 'Uso_Mensal_Horas', 'Tickets_Suporte', 'Churn_Status'].
    **Saída:** Tabela Markdown com a taxa de churn por cohort e um resumo executivo de 150 palavras destacando os 3 principais fatores de risco e o cohort mais problemático.
    ```

2.  **Criação de Segmentos de Risco (Creation of Risk Segments)**
    ```
    **Contexto:** Somos uma plataforma SaaS B2B. O churn é definido como a não renovação após 12 meses.
    **Tarefa:** Crie 3 segmentos de clientes de alto risco de churn (ex: 'Risco Extremo', 'Risco Moderado', 'Risco Latente') com base nas seguintes métricas: [Baixa Frequência de Login, Queda de 50% no Uso de Funcionalidades Chave, Abertura de 3+ Tickets de Suporte nos Últimos 30 Dias].
    **Saída:** Para cada segmento, forneça uma descrição, o critério de pontuação de risco e 2 ações de retenção específicas.
    ```

3.  **Análise de Sentimento de Feedback de Churn (Churn Feedback Sentiment Analysis)**
    ```
    **Tarefa:** Analise os seguintes 10 comentários de clientes que cancelaram e categorize o sentimento (Negativo, Neutro, Positivo) e o motivo principal (Preço, Funcionalidade, Suporte, Concorrência).
    **Dados:** [Lista de 10 comentários de feedback de cancelamento].
    **Saída:** Tabela com 'Comentário', 'Sentimento' e 'Motivo Principal'. Em seguida, sugira uma mudança de produto ou processo para mitigar o motivo mais frequente.
    ```

4.  **Simulação de Modelo Preditivo (Predictive Model Simulation)**
    ```
    **Papel:** Machine Learning Engineer.
    **Tarefa:** Simule um modelo de classificação (ex: Random Forest) para prever o churn.
    **Features:** [Idade do Cliente, Tempo de Contrato, Valor do Contrato (MRR), Número de Logins/Semana, Uso da Feature X, Uso da Feature Y].
    **Saída:** Explique quais seriam as 3 features mais importantes para a previsão e por que, com base no conhecimento de mercado. Crie um prompt de acompanhamento para refinar a análise.
    ```

5.  **Criação de Playbook de Retenção (Retention Playbook Creation)**
    ```
    **Contexto:** O cliente 'ID 456' está no segmento 'Risco Extremo' (Queda de 70% no uso e 0 interações com o Suporte).
    **Tarefa:** Crie um *playbook* de retenção de 3 etapas para o Customer Success Manager (CSM) usar.
    **Etapas:** 1. Contato Inicial (Canal e Mensagem), 2. Oferta de Valor (Incentivo), 3. Acompanhamento (Próxima Ação).
    **Saída:** Um script detalhado para a Etapa 1 (e-mail ou mensagem de chat) e a lógica por trás da Oferta de Valor.
    ```

6.  **Análise de Churn Competitivo (Competitive Churn Analysis)**
    ```
    **Tarefa:** Analise o seguinte feedback de clientes que migraram para o concorrente 'X'.
    **Feedback:** [Lista de 5 razões pelas quais os clientes foram para o concorrente X].
    **Saída:** Identifique o principal diferencial do concorrente X percebido pelos clientes e sugira 3 pontos de melhoria no nosso produto/serviço para neutralizar essa vantagem.
    ```
```

## Best Practices
**Melhores Práticas (Best Practices)**:
1.  **Fornecer Contexto e Dados Estruturados**: Sempre inclua o máximo de dados relevantes (ex: CSV, JSON, ou descrição detalhada do dataset) e o contexto de negócios (ex: tipo de produto, período de análise, definição de churn).
2.  **Definir o Papel (Role-Playing)**: Comece o prompt definindo o LLM como um "Cientista de Dados Sênior", "Analista de Sucesso do Cliente" ou "Especialista em Retenção" para garantir uma resposta com a perspectiva correta.
3.  **Especificar a Saída (Output Structuring)**: Peça a saída em um formato específico (ex: tabela Markdown, JSON, resumo executivo de 150 palavras) para facilitar a análise e integração em relatórios.
4.  **Análise de Causa Raiz (Root Cause Analysis)**: Não se limite a pedir a previsão; peça a análise dos **fatores preditivos** e a **justificativa** para a pontuação de risco de churn.
5.  **Iteração e Refinamento**: Use a saída inicial para prompts de acompanhamento, como "Com base nos 3 principais fatores de risco, crie 5 ações de retenção personalizadas para o segmento 'Usuários de Baixa Atividade'".

**Best Practices (English)**:
1.  **Provide Context and Structured Data**: Always include as much relevant data as possible (e.g., CSV, JSON, or detailed dataset description) and the business context (e.g., product type, analysis period, churn definition).
2.  **Define the Role (Role-Playing)**: Start the prompt by defining the LLM as a "Senior Data Scientist," "Customer Success Analyst," or "Retention Specialist" to ensure the response has the correct perspective.
3.  **Specify the Output (Output Structuring)**: Ask for the output in a specific format (e.g., Markdown table, JSON, 150-word executive summary) to facilitate analysis and integration into reports.
4.  **Root Cause Analysis**: Don't just ask for the prediction; ask for the analysis of the **predictive factors** and the **justification** for the churn risk score.
5.  **Iteration and Refinement**: Use the initial output for follow-up prompts, such as "Based on the top 3 risk factors, create 5 personalized retention actions for the 'Low Activity Users' segment."

## Use Cases
**Casos de Uso (Use Cases)**:
1.  **Segmentação de Clientes em Risco**: Identificar grupos de clientes com alta probabilidade de cancelamento para campanhas de retenção direcionadas.
2.  **Otimização de Recursos de Suporte**: Analisar tickets de suporte e interações para identificar padrões de insatisfação que precedem o churn, permitindo a intervenção proativa.
3.  **Validação de Hipóteses de Produto**: Usar o LLM para analisar feedback de cancelamento e validar se a falta de uma funcionalidade específica ou um problema de usabilidade está impulsionando o churn.
4.  **Criação de Conteúdo de Retenção**: Gerar rascunhos de e-mails, mensagens de chat ou ofertas personalizadas para clientes em risco, adaptadas ao seu perfil de uso e motivo de insatisfação.
5.  **Relatórios Executivos Rápidos**: Transformar dados brutos ou resumos de modelos de Machine Learning em um resumo executivo claro e conciso para a liderança, economizando tempo do analista.
6.  **Análise de Churn por Categoria**: Analisar o churn em diferentes linhas de produto ou serviços para entender onde o problema é mais agudo e por quê.

## Pitfalls
**Armadilhas Comuns (Common Pitfalls)**:
1.  **Injeção de Dados Brutos Excessivos**: Tentar inserir um arquivo CSV inteiro (milhares de linhas) diretamente no prompt. LLMs têm limites de contexto e podem falhar ou gerar resultados imprecisos. **Solução**: Fornecer amostras representativas ou apenas a descrição estatística dos dados.
2.  **Falta de Definição de Churn**: Não definir claramente o que constitui "churn" para o negócio (ex: cancelamento imediato, não renovação, inatividade por 90 dias). Isso leva a análises vagas.
3.  **Viés de Confirmação**: Pedir ao LLM para confirmar uma hipótese pré-existente (ex: "O preço é o principal motivo do churn, certo?"). O LLM pode apenas regurgitar a hipótese em vez de realizar uma análise objetiva.
4.  **Ignorar o Papel do LLM**: Tratar o LLM como um software de estatística que executa código. O LLM é um **mecanismo de raciocínio e linguagem**. Ele deve ser usado para **interpretar** dados e **gerar estratégias**, não para cálculos estatísticos complexos que exigem ferramentas como Python/Pandas.
5.  **Saída Não Estruturada**: Não especificar o formato de saída. Isso resulta em longos blocos de texto difíceis de digerir e usar em relatórios de negócios.

## URL
[https://pt.linkedin.com/pulse/guia-pr%C3%A1tico-de-engenharia-prompt-do-b%C3%A1sico-ao-adrianno-esnarriaga-polqf](https://pt.linkedin.com/pulse/guia-pr%C3%A1tico-de-engenharia-prompt-do-b%C3%A1sico-ao-adrianno-esnarriaga-polqf)
