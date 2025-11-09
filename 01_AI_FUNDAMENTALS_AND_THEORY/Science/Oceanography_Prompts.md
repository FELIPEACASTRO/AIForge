# Oceanography Prompts

## Description
**Oceanography Prompts** referem-se a instruções de linguagem natural (prompts) projetadas para interagir com Modelos de Linguagem Grande (LLMs) especializados ou adaptados para o domínio da **Ciência Oceânica** (Oceanografia). Ao contrário de prompts genéricos, estes são formulados para tarefas complexas como análise de dados oceanográficos, modelagem preditiva, simulação de fenômenos marinhos, e recuperação de informações altamente específicas de vastos corpora de literatura científica e conjuntos de dados.

O desenvolvimento de modelos como o **OceanGPT** (o primeiro LLM para o domínio oceânico) e plataformas como o **OceanAI** demonstram a necessidade de prompts que incorporem conhecimento especializado, terminologia técnica (e.g., "reanálise CORA", "depósitos pelágicos"), e requisitos de formato de saída para análise científica (e.g., código, visualizações de dados).

Uma estrutura de prompt complementar, o processo **OCEAN** (Objective, Context, Examples, Assess, Negotiate), também é relevante como uma **melhor prática** para refinar a interação com qualquer LLM, garantindo que os resultados sejam focados, contextualmente precisos e cientificamente rigorosos.

## Examples
```
1.  **Consulta de Dados Específicos:**
    `"Mostre o nível da água em Boston a partir da reanálise CORA em junho de 1993. Gere o resultado como um gráfico de linha temporal e forneça o código Python usado para a visualização."`

2.  **Análise de Fenômenos:**
    `"Explique o mecanismo de formação e dissipação de um El Niño de intensidade moderada. Inclua os principais indicadores oceanográficos (SST, termoclina) e sugira um modelo de previsão de curto prazo."`

3.  **Simulação de Robótica Subaquática:**
    `"Simule o planejamento de trajetória para um Veículo Subaquático Autônomo (AUV) para mapear um recife de coral de 500m x 500m. O AUV deve manter uma altitude de 5 metros acima do fundo do mar. Gere o código de simulação em Python usando a biblioteca 'auv_toolkit'."`

4.  **Revisão de Literatura Científica:**
    `"Realize uma revisão sistemática sobre o impacto da acidificação dos oceanos na calcificação de moluscos nos últimos 5 anos (2020-2025). Liste os 5 artigos mais citados e resuma suas principais conclusões."`

5.  **Geração de Conteúdo Educacional:**
    `"Crie um prompt para um LLM genérico que o instrua a atuar como um oceanógrafo. O objetivo é gerar um plano de aula de 45 minutos para alunos do ensino médio sobre a circulação termohalina, incluindo uma atividade prática e perguntas de avaliação."`

6.  **Modelagem Preditiva:**
    `"Com base nos dados de temperatura da superfície do mar (TSM) do Atlântico Tropical Sul nos últimos 10 anos, preveja a probabilidade de formação de um ciclone tropical na próxima temporada. Justifique a previsão com os índices climáticos relevantes."`

7.  **Identificação de Espécies:**
    `"Descreva as características morfológicas e o habitat da espécie *Vampyroteuthis infernalis* (Lula-vampiro). Crie um prompt de imagem para gerar uma representação fotorrealista da criatura em seu ambiente natural."`
```

## Best Practices
*   **Seja Específico e Técnico:** Use a terminologia oceanográfica correta (e.g., "batimetria", "corrente de contorno ocidental", "fitoplâncton diatomáceas") para refinar a busca do modelo em seu conhecimento especializado.
*   **Defina o Formato de Saída:** Para tarefas científicas, especifique o formato desejado (e.g., "Gere o resultado como código Python", "Tabela Markdown", "Resumo em formato IMRaD").
*   **Cite a Fonte de Dados (se aplicável):** Quando possível, inclua a fonte de dados ou o conjunto de dados a ser consultado (e.g., "dados do Argo float", "reanálise CORA", "imagens de satélite MODIS").
*   **Use o Framework OCEAN:** Aplique a estrutura **O**bjective, **C**ontext, **E**xamples, **A**ssess, **N**egotiate para refinar a interação, especialmente para tarefas complexas ou de alta precisão.
*   **Validação Humana:** Sempre trate a saída do LLM como um ponto de partida. A validação e interpretação humana dos resultados são cruciais na pesquisa científica.

## Use Cases
*   **Análise e Visualização de Dados:** Consultar grandes conjuntos de dados oceanográficos (temperatura, salinidade, correntes) usando linguagem natural e gerar visualizações ou scripts de análise.
*   **Revisão e Síntese de Literatura:** Acelerar a pesquisa científica, resumindo artigos, identificando tendências e comparando metodologias em publicações oceanográficas.
*   **Modelagem e Previsão:** Auxiliar na criação de modelos preditivos para marés, ondas, circulação oceânica, e eventos climáticos extremos.
*   **Educação e Treinamento:** Gerar materiais didáticos, quizzes e simulações interativas para estudantes de oceanografia.
*   **Robótica Marinha:** Simular e planejar missões para AUVs e ROVs (Veículos Operados Remotamente), incluindo otimização de rotas e detecção de anomalias.

## Pitfalls
*   **Confiança Excessiva em LLMs Genéricos:** Modelos não especializados (como GPT-4 ou Gemini genéricos) podem "alucinar" dados científicos ou terminologia técnica, levando a resultados incorretos.
*   **Falta de Contexto Científico:** Prompts muito vagos ou sem contexto técnico suficiente resultarão em respostas superficiais ou irrelevantes para a pesquisa oceanográfica.
*   **Ignorar a Validação de Dados:** Aceitar a saída do LLM sem verificar a fonte dos dados ou a validade científica do modelo/previsão gerada.
*   **Limitações de Dados em Tempo Real:** A maioria dos LLMs é treinada em dados históricos. Consultas sobre condições oceânicas em tempo real ou muito recentes podem falhar ou ser imprecisas.
*   **Não Especificar Unidades e Escalas:** Falhar em definir unidades de medida (e.g., Celsius vs. Kelvin, metros vs. pés) ou escalas espaciais/temporais pode levar a erros de interpretação.

## URL
[https://oceangpt.blue/oceangpt-en/](https://oceangpt.blue/oceangpt-en/)
