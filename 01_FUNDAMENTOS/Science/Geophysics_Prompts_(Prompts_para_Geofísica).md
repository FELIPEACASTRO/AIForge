# Geophysics Prompts (Prompts para Geofísica)

## Description
Prompts para Geofísica referem-se à aplicação de técnicas de **Prompt Engineering** para interagir com Grandes Modelos de Linguagem (LLMs) e Modelos de Fundação (FMs) no domínio das geociências, com foco particular na geofísica (sismologia, gravimetria, magnetometria, etc.). Esta área emergente, com publicações significativas a partir de 2024-2025, visa automatizar tarefas complexas, gerar dados sintéticos, interpretar dados geofísicos e auxiliar na pesquisa. O cerne da técnica reside em fornecer contexto, linguagem especializada e instruções estruturadas para que o LLM atue como um "agente geofísico" capaz de realizar análises de dados, gerar código, interpretar resultados e até mesmo auxiliar na tomada de decisões em fluxos de trabalho sísmicos inteligentes. A eficácia dos prompts em geofísica depende da inclusão de terminologia técnica (ex: "análise de atributos sísmicos", "inversão de dados de poço"), especificações de formato (ex: JSON, código Python) e a definição clara do papel do modelo (ex: "Atue como um geofísico de exploração").

## Examples
```
1.  **Interpretação Sísmica (Prompt de Agente):**
    "Atue como um geofísico de interpretação. Analise o seguinte conjunto de dados sísmicos (fornecer link ou descrição de atributos) e identifique as principais falhas e horizontes. Gere um resumo em Markdown e, em seguida, escreva o código Python (usando a biblioteca 'segyio') necessário para visualizar a seção sísmica, destacando as anomalias de amplitude que podem indicar hidrocarbonetos."

2.  **Geração de Dados Sintéticos (Prompt Estruturado):**
    "Gere 100 registros de dados sísmicos sintéticos em formato JSON. Cada registro deve incluir os seguintes campos: 'Localização' (coordenadas geográficas), 'Profundidade' (em metros), 'Velocidade de Onda P' (em m/s), 'Impedância Acústica' (em kg/m²s), e 'Litologia' (ex: Arenito, Xisto, Calcário). Assegure que a correlação entre 'Velocidade de Onda P' e 'Litologia' seja geologicamente plausível."

3.  **Análise de Atributos Sísmicos (Prompt de Análise):**
    "Com base nos princípios da geofísica de exploração, explique a relação entre o atributo sísmico 'Coerência' e a presença de falhas geológicas. Em seguida, forneça um prompt otimizado para um LLM que solicite a análise de um mapa de coerência para identificar descontinuidades estruturais, especificando a saída como uma lista de coordenadas e a classificação da falha (normal, inversa, de empurrão)."

4.  **Revisão de Literatura (Prompt de Pesquisa):**
    "Realize uma revisão concisa da literatura sobre a aplicação de Redes Neurais Convolucionais (CNNs) na detecção de domos de sal em dados sísmicos 3D. Identifique os três artigos mais relevantes publicados entre 2023 e 2025 e resuma a principal contribuição de cada um em uma tabela Markdown com colunas para 'Título', 'Autor Principal' e 'Metodologia Principal'."

5.  **Otimização de Parâmetros (Prompt de Fluxo de Trabalho):**
    "Você é um especialista em processamento sísmico. Descreva o fluxo de trabalho ideal para a migração pré-empilhamento em profundidade (PSDM) de um conjunto de dados marinhos. Para cada etapa (ex: picking de velocidade, modelagem de velocidade), forneça um prompt que um LLM poderia usar para otimizar um parâmetro específico (ex: 'Sugira um valor inicial de velocidade RMS para a camada de água, dada a profundidade média de 500m')."

6.  **Interpretação de Dados de Poço (Prompt Multimodal/Integração):**
    "Atue como um geofísico de poço. Integre os dados de perfilagem (Gamma Ray, Resistividade, Densidade) com a interpretação sísmica. Qual é o prompt mais eficaz para um LLM multimodal que, ao receber o gráfico de perfilagem (imagem) e o log de litologia (texto), sugira a melhor amarração (well-tie) com o horizonte sísmico 'Top Reservoir'?"
```

## Best Practices
*   **Definição de Papel (Role-Playing):** Comece o prompt definindo o papel do LLM (ex: "Atue como um geofísico de exploração", "Você é um especialista em sismologia") para evocar conhecimento especializado.
*   **Especificidade Técnica:** Use a terminologia geofísica correta (ex: "tempo de trânsito", "migração Kirchhoff", "anomalia de Bouguer") para refinar a busca e a precisão da resposta.
*   **Estrutura de Saída:** Exija formatos de saída estruturados (JSON, CSV, código Python, tabela Markdown) para facilitar a integração com ferramentas de análise geofísica.
*   **Contexto de Dados:** Sempre que possível, forneça o contexto dos dados (tipo de levantamento, área de estudo, parâmetros-chave) ou solicite que o LLM simule um contexto realista.
*   **Cadeia de Pensamento (CoT):** Para tarefas complexas (ex: interpretação), instrua o LLM a usar o *Chain-of-Thought* (CoT) ou a decompor o problema em etapas geofísicas lógicas antes de fornecer a resposta final.

## Use Cases
*   **Geração de Dados Sintéticos:** Criação de grandes conjuntos de dados de treinamento para modelos de Machine Learning em geofísica (ex: detecção de falhas, classificação de fácies) quando os dados reais são escassos.
*   **Automação de Fluxos de Trabalho:** Automatização de etapas repetitivas em fluxos de trabalho sísmicos (ex: controle de qualidade, picking de horizontes, otimização de parâmetros de processamento).
*   **Interpretação e Análise:** Auxílio na interpretação de dados sísmicos, gravimétricos e magnéticos, identificando padrões e anomalias que podem ser difíceis de detectar manualmente.
*   **Pesquisa e Educação:** Síntese rápida de literatura científica, explicação de conceitos geofísicos complexos e geração de material didático específico.
*   **Monitoramento de Desastres:** Uso de LLMs multimodais para integrar dados de satélite (imagens) e relatórios textuais para avaliação rápida de danos após eventos geológicos (ex: terremotos, deslizamentos).

## Pitfalls
*   **Alucinações Geologicamente Implausíveis:** O LLM pode gerar dados sintéticos ou interpretações que violam princípios físicos ou geológicos fundamentais (ex: velocidades sísmicas irrealistas). **Mitigação:** Incluir restrições físicas e geológicas no prompt.
*   **Dependência Excessiva de Dados de Treinamento:** O LLM pode refletir vieses ou limitações dos dados de treinamento, especialmente se estes não incluírem dados geofísicos de alta qualidade. **Mitigação:** Sempre validar a saída do LLM com o conhecimento especializado humano.
*   **Prompts Vagos:** Prompts genéricos (ex: "Fale sobre sismologia") resultam em respostas superficiais e inutilizáveis para aplicações geofísicas. **Mitigação:** Sempre ser específico sobre o método, o dado e o resultado esperado.
*   **Ignorar a Natureza Multimodal:** A geofísica é inerentemente multimodal (dados sísmicos, logs de poço, mapas). Falhar em integrar ou referenciar a necessidade de análise de diferentes tipos de dados no prompt limita a utilidade do LLM.

## URL
[https://pubs.geoscienceworld.org/seg/tle/article/44/2/142/651624/Intelligent-seismic-workflows-The-power-of](https://pubs.geoscienceworld.org/seg/tle/article/44/2/142/651624/Intelligent-seismic-workflows-The-power-of)
