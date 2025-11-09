# Data Science Prompts

## Description
Prompts de Ciência de Dados são instruções estruturadas e detalhadas, fornecidas a Modelos de Linguagem Grande (LLMs), para auxiliar em todas as etapas do ciclo de vida da Ciência de Dados e Machine Learning (ML). Isso inclui desde a limpeza e pré-processamento de dados, Análise Exploratória de Dados (EDA), Engenharia de Features, construção e depuração de modelos, até a comunicação e interpretação de resultados para stakeholders. A eficácia desses prompts reside na capacidade de transformar tarefas complexas e demoradas em comandos claros, aproveitando o LLM como um assistente de codificação, análise e comunicação. A técnica visa aumentar a produtividade, automatizar tarefas repetitivas e garantir que as análises e os resultados do modelo sejam traduzidos em insights de negócios acionáveis.

## Examples
```
**1. Limpeza e Pré-processamento de Dados (Intermediário)**
```
## System
Você é um assistente de limpeza de dados experiente.

## User
Aqui está o resumo do meu DataFrame Pandas (inclua o output de df.info() e df.describe()).

## Task
1. Identifique colunas com mais de 20% de valores ausentes.
2. Sugira a melhor estratégia de imputação para as colunas numéricas e categóricas.
3. Forneça o código Python (usando pandas ou scikit-learn) para executar a limpeza e imputação, explicando cada passo.
```

**2. Análise Exploratória de Dados (EDA) (Intermediário)**
```
## System
Você é um narrador de análise de dados com foco em tendências de negócios.

## User
Tenho um conjunto de dados de vendas com as colunas: 'data', 'id_produto', 'regiao', 'unidades_vendidas', 'preco'.

## Task
Crie um checklist de EDA para examinar sazonalidade, outliers e tendências de vendas. Inclua o código Python para:
1. Um gráfico de linha para vendas ao longo do tempo.
2. Um boxplot de 'unidades_vendidas' por 'regiao'.
3. Interprete os resultados de cada visualização em termos de impacto no negócio.
```

**3. Engenharia de Features (Avançado)**
```
## System
Você é um Engenheiro de Features de Machine Learning.

## User
Estou construindo um modelo de regressão para prever o preço de imóveis. As colunas disponíveis são: 'area_quadrada', 'quartos', 'banheiros', 'ano_construcao', 'bairro'.

## Task
1. Sugira 5 features derivadas que podem melhorar a performance do modelo (ex: 'idade_imovel').
2. Escreva o código Python usando pandas para criar essas 5 features.
3. Gere uma classe Scikit-learn Transformer para encapsular essas transformações.
```

**4. Interpretação de Modelo (Comunicação Executiva)**
```
## System
Você é um contador de histórias de dados sênior, especializado em comunicação executiva.

## User
Aqui estão os valores SHAP (feature, impacto): [('renda', 0.45), ('idade', 0.20), ('historico_credito', 0.15), ('divida', 0.10)].

## Task
1. Classifique os 3 principais fatores de risco por impacto absoluto.
2. Escreva um resumo de ~120 palavras para o Conselho de Administração, explicando o que aumenta e o que reduz o risco.
3. Sugira duas ações de mitigação concretas.

## Constraints & Style
- Público: Nível de Diretoria, não-técnico.
- Tom: Confiante e focado em insights.
- Formato: Lista de tópicos em Markdown.
```

**5. Depuração de Modelo (Overfitting)**
```
## System
Você é um especialista em Machine Learning.

## User
Meu modelo RandomForestClassifier do sklearn apresenta alta acurácia no treino (98%) e baixa acurácia na validação (75%).

## Task
1. Liste 3 razões prováveis para o overfitting.
2. Para cada razão, forneça uma sugestão de correção e o código Python correspondente (ex: ajuste de hiperparâmetros, validação cruzada).
3. Explique o conceito de "viés-variância" em termos simples.
```
```

## Best Practices
**1. Seja Específico e Estruturado:** Defina o papel do LLM (ex: "Você é um Engenheiro de Feature experiente"), forneça o contexto (esquema do DataFrame, problema de negócio) e use delimitadores claros (como `###` ou `##`).
**2. Decomposição de Tarefas (Chaining):** Para fluxos de trabalho complexos (limpeza, EDA, modelagem), divida a tarefa em prompts modulares e encadeados, onde a saída de um prompt serve como entrada para o próximo.
**3. Exija o Código e a Explicação:** Peça explicitamente pelo código (Python, SQL, R) e por uma explicação linha a linha ou um resumo do raciocínio por trás da solução.
**4. Defina o Público e o Tom:** Ao solicitar relatórios ou resumos de resultados, especifique o público-alvo (ex: "Executivos não-técnicos", "Cientistas de Dados Juniores") e o tom desejado (ex: "Conciso e focado em custo", "Didático e detalhado").
**5. Use Few-Shot Learning:** Inclua exemplos de entrada e saída desejadas para guiar o modelo, especialmente para tarefas de formatação ou transformação de dados.

## Use Cases
**1. Automação de EDA e Limpeza de Dados:** Gerar código para identificar e tratar valores ausentes, outliers e inconsistências de formato em grandes conjuntos de dados.
**2. Engenharia de Features Acelerada:** Criar features derivadas complexas (ex: variáveis de defasagem de tempo, codificação de alta cardinalidade) e encapsulá-las em classes reutilizáveis.
**3. Explicação de Métricas de Modelo:** Traduzir métricas complexas (ex: Matriz de Confusão, F1-Score) em termos de impacto financeiro ou operacional para o público de negócios.
**4. Storytelling de Dados:** Transformar resultados técnicos de modelos (ex: valores SHAP, coeficientes de regressão) em narrativas concisas e acionáveis para relatórios executivos.
**5. Geração de Dados Sintéticos:** Criar amostras de dados sintéticos que imitam a distribuição e as características de um conjunto de dados real para fins de teste e desenvolvimento.
**6. Depuração e Otimização de Modelos:** Diagnosticar problemas comuns de ML (overfitting, underfitting, desvio de conceito) e sugerir soluções de código para correção.

## Pitfalls
**1. Linguagem Excessivamente Técnica:** Usar jargões de ML (ex: "AUC", "ROC", "SHAP") ao se comunicar com stakeholders não-técnicos. O prompt deve exigir a tradução para termos de negócio (ex: "custo de falso positivo").
**2. Falta de Contexto:** Não fornecer o esquema do conjunto de dados, o problema de negócio ou o formato de entrada/saída esperado. O LLM pode gerar código ou análises irrelevantes.
**3. Confiança Cega no Código:** Aceitar o código gerado sem revisão. O LLM pode cometer erros sutis de lógica ou usar bibliotecas desatualizadas. O prompt deve incluir uma etapa de "auto-avaliação" ou "revisão de código".
**4. Prompts de 'Caixa Preta':** Pedir apenas o resultado final sem exigir o raciocínio (Chain-of-Thought). Isso dificulta a depuração e a compreensão do processo de análise.
**5. Não Especificar o Formato de Saída:** Se o resultado for usado em um pipeline automatizado, a falta de especificação de formato (JSON, CSV, código Python) pode quebrar o fluxo de trabalho.

## URL
[https://towardsdatascience.com/the-end-to-end-data-scientists-prompt-playbook/](https://towardsdatascience.com/the-end-to-end-data-scientists-prompt-playbook/)
