# Research Methodology Prompts

## Description
A técnica de **Prompts de Metodologia de Pesquisa** (Research Methodology Prompts) consiste em estruturar comandos para modelos de linguagem (LLMs) com o objetivo de auxiliar em todas as etapas do processo de pesquisa científica, desde o planejamento inicial até a análise de dados. Em vez de comandos genéricos, esses prompts são projetados para atribuir ao LLM um **papel específico** (ex: consultor metodológico, designer experimental, estatístico) e fornecer **contexto e restrições detalhadas** sobre o objetivo da pesquisa, variáveis, população e tipo de estudo. O foco é transformar o LLM em um assistente de pesquisa que ajuda a mapear a metodologia, refinar questões, desenhar protocolos e planejar a análise, garantindo que a base do trabalho seja sólida e bem justificada.

## Examples
```
1. **Seleção de Método**: "Você é um consultor metodológico. Com base no objetivo de pesquisa: [Objetivo], sugira o método mais apropriado (quantitativo, qualitativo ou misto). Justifique sua escolha e explique como ela atende ao objetivo."
2. **Desenho Experimental**: "Aja como um designer de pesquisa. Desenvolva um protocolo experimental detalhado para o estudo descrito: [Descrição do Estudo]. Inclua condições de controle, variáveis, estratégia de randomização e plano de medição."
3. **Questões de Pesquisa**: "Crie um conjunto de 10 perguntas de pesquisa para um questionário alinhado com o objetivo: [Objetivo]. Garanta clareza, neutralidade e sugira o tipo de escala de resposta mais adequado (ex: Likert de 5 pontos)."
4. **Estratégia de Amostragem**: "Com base na população [População-alvo] e no objetivo [Objetivo], recomende uma estratégia de amostragem. Inclua o tamanho da amostra sugerido, o método de seleção (ex: estratificada, por conveniência) e a justificativa."
5. **Limitações Metodológicas**: "Identifique e explique três potenciais limitações metodológicas no estudo [Descrição do Estudo]. Para cada uma, sugira uma estratégia de mitigação ou como elas devem ser abordadas na discussão."
6. **Análise Estatística**: "Você é um consultor estatístico. Dado o desenho do estudo [Desenho], as variáveis [Variáveis] e o tamanho da amostra [Tamanho], recomende os testes estatísticos mais apropriados (ex: ANOVA, Regressão, Teste T) e justifique a escolha."
7. **Framework de Codificação Qualitativa**: "Com base nas seguintes perguntas de pesquisa qualitativa [Perguntas] e em uma amostra de dados [Amostra de Dados], gere um framework de codificação temática. Inclua temas principais, subtemas e exemplos de códigos."
```

## Best Practices
1.  **Definir o Papel e o Objetivo**: Comece atribuindo um papel profissional ao LLM (ex: "Você é um consultor estatístico") e vincule o prompt a um objetivo de pesquisa claro e específico.
2.  **Adicionar Contexto e Restrições**: Sempre inclua o máximo de contexto possível, como variáveis, tamanho da amostra, tipo de estudo (quantitativo, qualitativo, misto) e público-alvo. Prompts vagos resultam em respostas vagas.
3.  **Estruturar a Saída**: Peça ao LLM para estruturar a resposta em formatos específicos (ex: tabela, lista numerada, protocolo passo a passo) para facilitar a revisão e aplicação.
4.  **Revisar e Refinar Criticamente**: Trate a saída do LLM como um primeiro rascunho ou sugestão. Use seu julgamento crítico para ajustar, polir ou retrabalhar o material, especialmente em questões de validade e nuances complexas.
5.  **Iterar e Testar**: O *prompting* é uma habilidade. Quanto mais você testa e ajusta seus prompts, mais valor você extrairá em cada fase da sua pesquisa.

## Use Cases
1.  **Seleção de Métodos**: Sugerir e justificar o método de pesquisa mais apropriado (quantitativo, qualitativo ou misto) com base no objetivo.
2.  **Desenho Experimental**: Desenvolver protocolos experimentais detalhados, incluindo condições de controle, variáveis, estratégias de randomização e planos de medição.
3.  **Instrumentos de Coleta**: Criar conjuntos de perguntas para questionários e entrevistas, garantindo clareza, neutralidade e escalas de resposta adequadas.
4.  **Estratégias de Amostragem**: Recomendar o tamanho da amostra, o método de seleção (ex: aleatória, estratificada) e a justificativa com base na população e no objetivo.
5.  **Análise de Dados**: Planejar análises estatísticas (recomendar testes e justificar) ou sugerir abordagens de análise qualitativa (ex: teoria fundamentada, análise narrativa).
6.  **Mitigação de Limitações**: Identificar potenciais limitações metodológicas e sugerir estratégias de mitigação para fortalecer a validade do estudo.
7.  **Desenvolvimento de Frameworks**: Gerar *frameworks* de codificação temática para análise de dados qualitativos.

## Pitfalls
1.  **Substituição do Pensamento Crítico**: Confiar no LLM para validar descobertas ou interpretar nuances complexas. O LLM é uma ferramenta de organização e sugestão, não um substituto para o rigor acadêmico e o julgamento do pesquisador.
2.  **Entrada Insuficiente**: Fornecer prompts curtos ou genéricos que não incluem o contexto, as restrições e os dados necessários, resultando em saídas superficiais ou irrelevantes.
3.  **Aceitar a Saída Final**: Tratar a resposta do LLM como o trabalho final. A saída deve ser sempre considerada um rascunho que exige revisão, validação e refinamento humano.

## URL
[https://askyourpdf.com/blog/chatgpt-prompts-for-research](https://askyourpdf.com/blog/chatgpt-prompts-for-research)
