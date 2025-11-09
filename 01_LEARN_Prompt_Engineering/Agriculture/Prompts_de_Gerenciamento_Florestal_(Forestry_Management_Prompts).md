# Prompts de Gerenciamento Florestal (Forestry Management Prompts)

## Description
**Prompts de Gerenciamento Florestal** são instruções de engenharia de prompt especificamente formuladas para interagir com Modelos de Linguagem Grande (LLMs) e outras ferramentas de Inteligência Artificial (IA) no contexto da silvicultura, ecologia e manejo de recursos naturais. Eles são projetados para traduzir a complexidade dos dados ambientais e dos objetivos de manejo florestal em comandos claros e estruturados que a IA pode processar. O objetivo principal é alavancar a IA para tarefas como análise de inventário florestal, interpretação de dados de sensoriamento remoto (como NDVI), planejamento de tratamentos silviculturais, modelagem de risco de incêndios e avaliação de saúde florestal. A eficácia desses prompts reside na sua capacidade de fornecer **contexto técnico detalhado**, **definir a estrutura dos dados de entrada** e **solicitar um formato de saída específico** e acionável, transformando a IA em uma poderosa ferramenta de apoio à decisão para profissionais florestais.

## Examples
```
**1. Análise de Inventário Florestal (CoT):**
"Eu tenho dados de inventário florestal para um povoamento de 50 hectares de Pinus elliottii. Os dados incluem as colunas: 'parcela_id', 'DAP_cm', 'Altura_m', 'Volume_m3'. Por favor, siga estes passos: 1. Calcule a média, mediana e desvio padrão do DAP e da Altura. 2. Estime o volume total de madeira no povoamento. 3. Sugira um plano de desbaste (percentual de remoção e árvores-alvo) para atingir uma densidade de 400 árvores/ha, justificando a decisão com base nos dados de distribuição de DAP."

**2. Interpretação de Sensoriamento Remoto:**
"Analise a seguinte série temporal de valores NDVI (Índice de Vegetação por Diferença Normalizada) para uma área de reflorestamento de Eucalipto (coordenadas: [LAT, LON]) de 2023 a 2025: [LISTA DE DATAS E VALORES NDVI]. Identifique padrões sazonais, detecte qualquer queda abrupta que possa indicar distúrbio (ex: incêndio ou praga) e correlacione os picos de vigor com os períodos de maior precipitação na região. Apresente a análise em formato de relatório conciso."

**3. Planejamento de Tratamento Silvicultural:**
"Eu sou um silvicultor planejando o manejo de um povoamento de Cedro Australiano (Cedrela odorata) de 15 anos. O objetivo é a produção de madeira de alto valor. As características do local são: Elevação: 850m, Solo: Argiloso, Densidade atual: 650 árvores/ha. Qual é a melhor estratégia de poda e desbaste para maximizar o crescimento em diâmetro e a qualidade do fuste? Apresente um cronograma de intervenções para os próximos 10 anos, incluindo a justificativa técnica para cada ação."

**4. Avaliação de Risco de Incêndio:**
"Atue como um especialista em modelagem de risco de incêndio florestal. Com base nas seguintes condições: Espécie dominante: Pinus taeda, Topografia: Encosta íngreme (35% de declividade), Umidade do combustível: 8%, Velocidade do vento: 25 km/h. Descreva o cenário de propagação de fogo mais provável (taxa de propagação, altura da chama) e sugira três medidas preventivas imediatas para a equipe de campo. Use o modelo de Rothermel como base para a análise."

**5. Otimização de Colheita:**
"Gere um código Python (usando a biblioteca Pandas) para processar um arquivo CSV chamado 'colheita.csv'. O arquivo tem colunas 'Coordenada_X', 'Coordenada_Y', 'Volume_m3' e 'Custo_Colheita_R$'. O objetivo é identificar as 10 parcelas com a melhor relação custo-benefício (Volume/Custo) para priorizar a colheita. O código deve carregar o arquivo, calcular a métrica e imprimir as 10 melhores parcelas em ordem decrescente."

**6. Interpretação de Resultados de Teste de Solo (Few-Shot):**
"Interprete os resultados de teste de solo para fins de reflorestamento. Exemplo de análise: Amostra A: pH 5.2, P 5 mg/dm3, K 40 mg/dm3. Análise: Solo moderadamente ácido, deficiente em Fósforo e Potássio. Requer calagem e adubação NPK 04-14-08. Agora, analise esta nova amostra: Amostra B: pH 6.5, P 15 mg/dm3, K 120 mg/dm3. Qual é a recomendação de correção e adubação para o plantio de Teca (Tectona grandis)?"
```

## Best Practices
**1. Contextualização Detalhada:** Sempre comece o prompt definindo claramente o contexto florestal (espécie, localização, tipo de manejo, dados de entrada). A precisão do resultado depende da precisão do contexto. **2. Definição de Termos Técnicos:** Evite ambiguidades. Se usar terminologia silvicultural ou ecológica específica (ex: "windthrow", "DBH", "NDVI"), defina-a brevemente, especialmente se o modelo de IA não for especializado. **3. Estrutura de Dados Clara:** Ao fornecer dados (tabelas, CSVs), descreva a estrutura (colunas, unidades de medida) para que a IA possa processá-los corretamente. **4. Saída Específica:** Solicite o formato de saída desejado (tabela, resumo, código Python, plano de tratamento) para garantir a utilidade da resposta. **5. Uso de Chain-of-Thought (CoT):** Para análises complexas (ex: cálculo de carbono, modelagem de risco), guie a IA com um processo passo a passo para melhorar a precisão e a rastreabilidade do raciocínio.

## Use Cases
nan

## Pitfalls
**1. Solicitações Vagas:** Pedir à IA para "falar sobre a saúde da floresta" sem especificar a região, espécie, patógeno ou período de tempo. A falta de especificidade leva a respostas genéricas e inúteis. **2. Falta de Contexto Geográfico/Ecológico:** Omitir informações cruciais como a zona biogeoclimática, tipo de solo ou histórico de manejo. A silvicultura é altamente dependente do local. **3. Assumir Conhecimento Técnico da IA:** Usar códigos de espécies regionais (ex: 'DF' para Douglas-fir) ou unidades de medida não padrão sem defini-los. **4. Sobrecarga de Dados Não Estruturados:** Tentar colar grandes blocos de dados brutos sem descrever a estrutura ou o que se espera da análise. **5. Ignorar a Necessidade de Fontes:** Usar a IA para obter dados factuais (ex: equações alométricas) sem solicitar a fonte ou a referência científica, o que pode levar a erros de cálculo críticos no manejo.

## URL
[https://aiforester.com/learning/prompt-engineering-for-forestry.html](https://aiforester.com/learning/prompt-engineering-for-forestry.html)
