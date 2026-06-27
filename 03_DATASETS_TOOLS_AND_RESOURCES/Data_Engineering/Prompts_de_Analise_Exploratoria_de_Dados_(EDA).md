# Prompts de Análise Exploratória de Dados (EDA)

## Description
Prompts de Análise Exploratória de Dados (EDA) são instruções estruturadas fornecidas a Modelos de Linguagem Grande (LLMs) para automatizar, guiar e acelerar o processo de EDA. A EDA é uma etapa fundamental na ciência de dados que visa resumir as principais características de um conjunto de dados, muitas vezes usando métodos visuais. Ao usar prompts, o analista pode delegar tarefas como limpeza de dados, cálculo de estatísticas descritivas, identificação de *outliers*, detecção de valores ausentes e sugestão de visualizações. O uso eficaz de prompts de EDA transforma o LLM em um assistente de dados interativo, permitindo que o analista se concentre na interpretação dos *insights* em vez da codificação repetitiva. A chave é fornecer contexto claro, definir a estrutura dos dados e solicitar uma saída acionável (código, tabelas ou resumos executivos).

## Examples
```
**1. Análise Estatística Descritiva Completa**
```
Aja como um Cientista de Dados Sênior. Analise o arquivo CSV 'vendas_mensais.csv'. Gere uma tabela Markdown que inclua: contagem de valores não nulos, média, desvio padrão, mínimo, máximo, Q1, Q2 (mediana) e Q3 para todas as colunas numéricas. Para colunas categóricas, liste a contagem de valores únicos e a moda.
```
**2. Identificação de Outliers e Valores Ausentes**
```
Com base no conjunto de dados 'dados_clientes.csv', identifique todas as colunas com mais de 5% de valores ausentes. Para a coluna 'Renda_Anual', use o método IQR para detectar e listar os 10 principais outliers. Sugira a melhor estratégia de imputação para os valores ausentes na coluna 'Idade'.
```
**3. Sugestão de Visualizações para Relações de Variáveis**
```
Meu objetivo é entender a relação entre 'Tempo_de_Serviço' (numérica) e 'Taxa_de_Churn' (binária) no 'dataset_telecom.csv'. Sugira 3 tipos de gráficos mais informativos para visualizar essa relação. Para cada gráfico, forneça o código Python (usando Matplotlib ou Seaborn) e a interpretação esperada.
```
**4. Análise de Distribuição e Normalidade**
```
Foque na coluna 'Preço_do_Imóvel' do 'dataset_imoveis.csv'. Descreva a forma da distribuição (simétrica, assimétrica à esquerda/direita). Calcule a curtose e a assimetria. Com base nesses resultados, o que você pode inferir sobre a normalidade dos dados?
```
**5. Segmentação e Comparação de Subgrupos**
```
Usando o 'dataset_marketing.csv', compare as métricas de 'Taxa_de_Conversão' e 'Custo_por_Aquisição' entre os subgrupos definidos pela coluna 'Canal_de_Marketing' (Email, Social, Search). Apresente os resultados em uma tabela comparativa e destaque o canal com o melhor ROI.
```
```

## Best Practices
**1. Fornecer Contexto e Estrutura:** Sempre defina a **Persona** (Ex: "Você é um Cientista de Dados Sênior"), o **Objetivo** (Ex: "Encontrar anomalias") e o **Formato de Saída** (Ex: "Tabela Markdown com 3 colunas: Variável, Estatística, Valor"). **2. Inserir Metadados do Conjunto de Dados:** Mencione o nome do conjunto de dados, o número de linhas/colunas e, crucialmente, liste as colunas relevantes e seus tipos de dados (categórico, numérico, temporal). **3. Iteração e Refinamento:** Comece com prompts amplos (análise estatística básica) e refine com prompts mais específicos (investigação de *outliers* em uma coluna específica) com base nas saídas anteriores (Chain-of-Thought para EDA). **4. Solicitar Código e Explicação:** Peça explicitamente o código (Python/R) usado para a análise, juntamente com uma explicação passo a passo dos resultados e das implicações para o negócio. **5. Gerenciamento de Dados Sensíveis:** Nunca insira dados confidenciais diretamente no prompt. Em vez disso, use amostras de dados sintéticos ou estatísticas resumidas, ou use ferramentas de IA que garantam a privacidade e o processamento local dos dados.

## Use Cases
**1. Limpeza e Pré-processamento de Dados:** Geração de código para padronizar formatos, lidar com valores ausentes (imputação) e corrigir erros tipográficos em grandes conjuntos de dados. **2. Geração de Estatísticas Descritivas:** Automatização do cálculo de métricas de tendência central, dispersão e forma para todas as variáveis de um *dataset*. **3. Criação de Visualizações:** Sugestão e geração de código para gráficos informativos (histogramas, *box plots*, gráficos de dispersão) para entender a distribuição e as relações entre variáveis. **4. Identificação de Anomalias e *Outliers*:** Uso de prompts para aplicar métodos estatísticos (como o escore Z ou IQR) para sinalizar pontos de dados incomuns que requerem investigação. **5. Resumo Executivo da EDA:** Síntese dos principais *insights* da análise em um formato de fácil compreensão para *stakeholders* não técnicos, destacando implicações de negócios. **6. Engenharia de Recursos (*Feature Engineering*):** Sugestão de novas variáveis ou transformações de dados que podem melhorar o desempenho de modelos de *Machine Learning* subsequentes.

## Pitfalls
**1. Confiança Excessiva na Saída:** Assumir que o código ou a análise do LLM está 100% correta sem validação. O LLM pode cometer erros estatísticos ou lógicos (*hallucinations*). **2. Prompts Vagos ou Incompletos:** Não fornecer o contexto de dados necessário (nomes de colunas, tipos de dados, objetivo da análise), resultando em saídas genéricas ou irrelevantes. **3. Ignorar a Estrutura do Prompt:** Não usar uma estrutura clara (Persona, Contexto, Tarefa, Formato), o que diminui a qualidade e a consistência da resposta. **4. Inserir Dados Confidenciais:** O risco de segurança e privacidade ao colar grandes volumes de dados sensíveis diretamente na interface do LLM. **5. Falha em Iterar:** Tratar o LLM como uma ferramenta de consulta única em vez de um parceiro interativo. A EDA é um processo iterativo, e os prompts devem refletir isso, refinando as perguntas com base nas descobertas anteriores.

## URL
[https://team-gpt.com/blog/chatgpt-prompts-for-data-analysis](https://team-gpt.com/blog/chatgpt-prompts-for-data-analysis)
