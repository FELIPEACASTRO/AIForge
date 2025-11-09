# Ecology Prompts (Prompts de Ecologia)

## Description
A técnica de engenharia de prompt focada na aplicação de Large Language Models (LLMs) em áreas relacionadas à **ecologia, conservação, gestão ambiental e sustentabilidade**. O termo "Ecology Prompts" abrange a criação de instruções que orientam a IA a processar dados ambientais, gerar relatórios de impacto, analisar estatísticas ecológicas e auxiliar na tomada de decisões sustentáveis.

O conceito se manifesta em duas vertentes principais:

1.  **Prompts para Aplicações Ecológicas:** O uso da IA como ferramenta para resolver problemas ambientais (por exemplo, monitoramento de biodiversidade, avaliação de impacto ambiental - AIA).
2.  **Green Prompting (Prompting Verde):** O uso de prompts para reduzir o consumo de energia e o impacto ambiental da própria IA, focando na eficiência e na minimização de tentativas e erros [2].

Essa técnica é crucial para alavancar o potencial da IA na pesquisa e gestão ambiental, garantindo que a complexidade dos dados ecológicos (como a estruturação espacial e temporal) seja tratada com o devido rigor [1].

## Examples
```
**1. Análise Estatística Ecológica (com CoT)**
`"Aja como um estatístico ecológico. Tenho um conjunto de dados de contagem de espécies (variável dependente) em 5 locais diferentes (variável categórica) ao longo de 10 anos (variável temporal).
1. Qual teste estatístico (e por que) é mais apropriado para analisar o efeito do local e do tempo na contagem de espécies, considerando a não-independência dos dados?
2. Gere o código R completo para executar este teste, incluindo a importação de um arquivo CSV chamado 'dados_ecologia.csv'.
3. Interprete os resultados de forma concisa, focando no p-valor e no tamanho do efeito."`

**2. Avaliação de Impacto Ambiental (AIA)**
`"Usando as diretrizes da US EPA 2025 para qualidade do ar, crie uma checklist detalhada para a fase de coleta de dados de uma Avaliação de Impacto Ambiental para a construção de uma nova rodovia. A checklist deve incluir parâmetros de monitoramento, frequência de coleta e fontes de dados recomendadas."`

**3. Planejamento de Conservação**
`"Identifique 5 estratégias inovadoras de planejamento de sítios para a criação de um corredor de vida selvagem em uma área urbana fragmentada. Baseie as estratégias no relatório 'Urban Eco 2025' e inclua a justificativa ecológica para cada uma."`

**4. Interpretação de Regulamentação**
`"Resuma as próximas regulamentações de proteção de zonas úmidas de 2025 e suas implicações diretas para o design de projetos de infraestrutura no estado de São Paulo. Apresente o resumo em formato de tabela com as colunas: Regulamento, Requisito Chave, Implicação no Projeto."`

**5. Geração de Conteúdo de Conscientização**
`"Escreva uma cópia clara e motivadora para uma campanha de conscientização sobre reciclagem, utilizando um tom de voz 'otimista e orientado para a ação'. O público-alvo são jovens adultos (18-25 anos). O texto deve ter no máximo 150 palavras e incluir uma chamada para ação (CTA)." [3]`

**6. Otimização de Prompt (Green Prompting)**
`"Reescreva o seguinte prompt para ser mais conciso e direto, mantendo a intenção original de gerar um resumo de 5 pontos sobre a energia solar fotovoltaica: 'Por favor, me forneça um resumo muito detalhado e abrangente, com pelo menos 5 pontos principais, sobre os benefícios e desafios da implementação de sistemas de energia solar fotovoltaica em larga escala em países em desenvolvimento.'"`

**7. Identificação de Tendências**
`"Quais são as três principais tendências em adoção de energia renovável a partir de 2023, conforme documentos de revisão de tecnologia e pesquisa pós-2023? Estruture a resposta com o nome da tendência, uma breve descrição e o impacto potencial no mercado." [3]`
```

## Best Practices
**1. Contextualização Ecológica Detalhada:** Inclua sempre o máximo de contexto específico da área (domínio) no prompt, como a espécie, o ecossistema, a localização geográfica e o período de tempo.
**2. Estrutura e Rigor Estatístico:** Ao solicitar análises estatísticas (como em ecologia), separe o fluxo de trabalho em componentes (por exemplo, "gerar código R para regressão", "interpretar resultados") e use técnicas como **Chain of Thought (CoT)** para forçar a IA a raciocinar passo a passo, garantindo o rigor [1].
**3. Especificação de Formato de Saída:** Defina claramente o formato desejado (tabela, checklist, código Python/R, relatório) para que a IA possa estruturar a resposta de forma útil.
**4. Uso de Referências Documentais:** Mencione documentos, relatórios ou diretrizes específicas (por exemplo, "Usando as diretrizes da US EPA 2025") para ancorar a resposta da IA em dados e regulamentações reais [3].
**5. Green Prompting (Eficiência):** Para reduzir o custo ambiental e computacional da IA, utilize prompts concisos e diretos. Evite prompts excessivamente longos ou ambíguos que exijam múltiplas iterações ou que resultem em respostas longas e desnecessárias [2].
**6. Supervisão Humana:** Mantenha a supervisão humana sobre as decisões estatísticas e as conclusões ambientais, pois os LLMs podem apresentar limitações de raciocínio em testes estatísticos complexos, especialmente aqueles com estruturação espacial e temporal [1].

## Use Cases
**1. Pesquisa e Análise Ecológica:** Auxiliar cientistas na escolha de modelos estatísticos apropriados para dados ecológicos complexos (com estruturação espacial e temporal) e na geração de código para análise de dados (R, Python) [1].
**2. Gestão Ambiental e Conformidade:** Criar checklists, resumir regulamentações (por exemplo, US EPA, leis locais) e gerar relatórios de conformidade para projetos de infraestrutura e desenvolvimento [3] [4].
**3. Avaliação de Impacto Ambiental (AIA):** Gerar listas de verificação para coleta de dados, identificar potenciais impactos na vida selvagem e sugerir estratégias de mitigação para projetos de construção [4].
**4. Conservação da Biodiversidade:** Automatizar a identificação de espécies a partir de imagens ou gravações de som (com ferramentas externas) e gerar planos de monitoramento e conservação [5].
**5. Comunicação e Conscientização:** Criar conteúdo envolvente e informativo para campanhas de sustentabilidade, relatórios corporativos de ESG (Ambiental, Social e Governança) e materiais educativos [3].
**6. Otimização de Custos e Sustentabilidade da IA (Green Prompting):** Aplicar a técnica para reduzir o consumo de energia dos LLMs, minimizando o número de tokens e a complexidade das inferências, alinhando a tecnologia com os objetivos de sustentabilidade [2].

## Pitfalls
**1. Falta de Contexto Ecológico:** Omitir detalhes cruciais (como a escala espacial/temporal, o tipo de dado ou a espécie) leva a respostas genéricas ou estatisticamente incorretas, especialmente em ecologia, onde a estruturação dos dados é vital [1].
**2. Confiança Cega em Análises Estatísticas:** LLMs podem gerar código estatístico (por exemplo, em R ou Python) que parece correto, mas que aplica o teste errado ou interpreta mal os resultados. A supervisão humana é indispensável para validar a escolha do modelo e a interpretação [1].
**3. Prompts Ambíguos ou Excessivamente Longos:** Prompts mal formulados ou muito extensos aumentam o número de tokens processados, elevando o consumo de energia e o custo computacional (o oposto do *Green Prompting*) [2].
**4. Ignorar a Fonte de Dados:** Solicitar análises ou relatórios sem especificar a fonte de dados (por exemplo, "Analise o impacto da poluição" sem fornecer o conjunto de dados) resulta em informações hipotéticas ou não verificáveis.
**5. Não Especificar o Papel (Persona):** Deixar de instruir a IA a agir como um "Cientista de Conservação" ou "Consultor Ambiental" pode reduzir a qualidade técnica e a precisão da resposta.

## URL
[https://ecoevorxiv.org/repository/view/9493/](https://ecoevorxiv.org/repository/view/9493/)
