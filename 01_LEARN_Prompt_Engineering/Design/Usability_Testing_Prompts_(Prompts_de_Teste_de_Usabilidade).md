# Usability Testing Prompts (Prompts de Teste de Usabilidade)

## Description
**Prompts de Teste de Usabilidade** (Usability Testing Prompts) são instruções de engenharia de prompt projetadas para alavancar Modelos de Linguagem Grande (LLMs), como ChatGPT ou Gemini, para automatizar, acelerar e aprimorar várias etapas do processo de Pesquisa de Experiência do Usuário (UX Research) e Teste de Usabilidade [1]. Em vez de substituir o pesquisador humano, esses prompts atuam como um **copiloto** ou **assistente**, ajudando a gerar rapidamente artefatos de pesquisa, como roteiros de entrevistas, perguntas de pesquisa, questionários de triagem (screener surveys), e até mesmo a analisar dados qualitativos e quantitativos [1].

A eficácia desses prompts reside na sua capacidade de transformar tarefas demoradas e repetitivas – como a redação de perguntas não tendenciosas ou a categorização de grandes volumes de feedback aberto – em saídas estruturadas e acionáveis. O uso de frameworks de prompting, como **REFINE**, **CARE** e **RACEF**, é fundamental para fornecer o contexto, as regras e o formato de saída necessários para que a IA produza resultados de alta qualidade e relevantes para o domínio de UX [1].

Em essência, a técnica permite que os pesquisadores de UX se concentrem na interpretação e na estratégia, enquanto a IA cuida da geração e do processamento inicial de dados, reduzindo significativamente o tempo de obtenção de *insights* (time-to-insight) [1].

## Examples
```
**1. Geração de Roteiro de Entrevista Semi-Estruturada (REFINE)**
```
**Papel:** Você é um pesquisador de UX sênior.
**Tarefa:** Crie um guia de entrevista semi-estruturada de 45 minutos para usuários avançados que cancelaram a assinatura do nosso produto SaaS de gestão de projetos nos últimos 3 meses.
**Formato:** O guia deve ser dividido em 5 seções cronometradas (Introdução, Contexto, Tarefas, Reflexão, Encerramento).
**Regras:** Inclua 7 perguntas centrais, com 2 perguntas de acompanhamento (follow-up) para cada uma. As perguntas centrais devem focar em **lacunas de recursos** e **gatilhos emocionais** que levaram ao cancelamento (churn).
```

**2. Verificação de Viés em Perguntas de Pesquisa (CARE)**
```
**Contexto:** Sou um pesquisador de UX preparando um questionário para avaliar a satisfação com o novo recurso 'Modo Escuro'.
**Pergunta (Ask):** Verifique as seguintes perguntas de pesquisa quanto a qualquer viés, perguntas indutoras (leading questions) ou irrelevância.
**Regras:** Se encontrar algum problema, forneça uma nova lista de perguntas objetivas e neutras.
**Exemplo de Entrada:** "Quão fácil foi para você usar o novo e intuitivo Modo Escuro?" e "Você concorda que o design melhorou com o Modo Escuro?"
```

**3. Criação de Questionário de Triagem (Screener Survey)**
```
**Tarefa:** Crie um questionário de triagem (screener survey) de 6 perguntas para recrutar participantes para um teste de usabilidade.
**Público-Alvo:** Gerentes de Produto (Product Managers) com pelo menos 3 anos de experiência no setor de FinTech que usam ativamente a ferramenta de análise 'Mixpanel' pelo menos 3 vezes por semana.
**Formato:** Forneça a pergunta, o formato de resposta (múltipla escolha, aberta) e a lógica de qualificação para cada pergunta.
```

**4. Análise Temática de Comentários Abertos (RACEF)**
```
**Foco:** Analise os 300 comentários abertos de NPS (Net Promoter Score) que estou fornecendo a seguir.
**Tarefa:** Agrupe os comentários em temas principais.
**Formato:** Retorne uma tabela com as seguintes colunas: 'Tema', 'Porcentagem de Comentários', e 'Duas Citações Representativas'.
**Regras:** Rotule cada tema de forma concisa e liste os temas em ordem decrescente de frequência.
**[Inserir 300 comentários aqui]**
```

**5. Síntese de Insights de Entrevistas**
```
**Contexto:** Abaixo estão as transcrições de 5 entrevistas semi-estruturadas sobre a funcionalidade de 'Upload de Documentos' do nosso aplicativo.
**Tarefa:** Sintetize os insights.
**Regras:** Codifique cada citação por **tema** e **sentimento** (positivo, negativo, neutro). Entregue uma lista classificada dos 5 principais problemas de usabilidade encontrados, incluindo citações ilustrativas para cada problema e um resumo de 100 palavras sobre a oportunidade de design.
**[Inserir Transcrições aqui]**
```

**6. Geração de Hipóteses Testáveis**
```
**Contexto:** O documento anexo (ou texto) destaca os principais pontos de fricção (friction points) no nosso funil de checkout.
**Tarefa:** Gere 5 hipóteses testáveis de design.
**Formato:** Cada hipótese deve seguir o formato: 'Acreditamos que [MUDANÇA] irá melhorar [MÉTRICA] para [USUÁRIO]'.
```

**7. Design de Pesquisa de Pulso (Pulse Survey)**
```
**Tarefa:** Desenhe uma pesquisa de pulso (pulse survey) de 8 perguntas para medir a satisfação do usuário com o recurso 'Busca por Voz' recém-lançado.
**Regras:** Misture 6 itens de escala Likert de 5 pontos cobrindo [Precisão, Velocidade, Facilidade de Uso] com 1 pergunta NPS. Mantenha o tempo de conclusão abaixo de 2 minutos.
```
```

## Best Practices
**1. Adote um Framework de Prompting (REFINE, CARE, RACEF):** Estruture seus prompts usando frameworks como REFINE (Role, Expectation, Format, Iterate, Nuance, Example), CARE (Context, Ask, Rules, Examples) ou RACEF (Rephrase, Append, Clarify, Examples, Focus) para garantir que o LLM tenha o contexto, as regras e o formato de saída desejados [1].
**2. Defina o Papel e o Objetivo:** Comece o prompt definindo claramente o papel da IA (ex: "Você é um pesquisador de UX sênior") e o objetivo específico da tarefa (ex: "Gerar um guia de discussão para teste de usabilidade") [1].
**3. Seja Específico e Contextual:** Forneça o máximo de detalhes e contexto possível. Inclua o produto, o público-alvo, o tempo de duração, o formato de saída (tabela, lista, resumo) e quaisquer restrições [1].
**4. Peça por Não-Viés:** Inclua uma instrução explícita para que a IA verifique e remova qualquer **viés cognitivo** ou **perguntas indutoras** (leading questions) nos resultados gerados, especialmente em perguntas de entrevista ou pesquisa [1].
**5. Itere e Refine:** O primeiro resultado da IA raramente é o final. Use prompts de acompanhamento para iterar, refinar, adicionar nuances ou remover seções desnecessárias, seguindo o princípio do "Iterate" do framework REFINE [1].
**6. Use a IA para Análise de Dados:** Utilize prompts para tarefas de análise de dados qualitativos e quantitativos, como agrupar comentários abertos por tema, identificar pontos de abandono em funis (drop-off) ou sintetizar transcrições de entrevistas [1].
**7. Peça por Citações e Referências:** Sempre que possível, peça à IA para citar as fontes ou fornecer referências para suas sugestões, e **sempre verifique** essas fontes para garantir a precisão e evitar alucinações [1].

## Use Cases
**1. Geração de Artefatos de Pesquisa:**
*   **Roteiros de Entrevista:** Criação rápida de guias de discussão semi-estruturados, incluindo perguntas centrais, follow-ups e estrutura de tempo [1].
*   **Questionários de Pesquisa (Surveys):** Design de questionários com diferentes tipos de perguntas (Likert, abertas, fechadas), garantindo a neutralidade e o foco em hipóteses específicas [1].
*   **Questionários de Triagem (Screeners):** Elaboração de perguntas para recrutar perfis de participantes muito específicos (ex: Gerentes de Produto com X anos de experiência em Y setor) [1].
*   **Checklists de Moderação:** Geração de listas de verificação abrangentes para moderadores de entrevistas, cobrindo aspectos técnicos, construção de *rapport* e anotações [1].

**2. Análise e Síntese de Dados:**
*   **Análise Temática Qualitativa:** Agrupamento e rotulagem de grandes volumes de dados abertos (ex: comentários de NPS, feedback de usuários) em temas acionáveis, com cálculo de frequência e citações representativas [1].
*   **Identificação de Pontos de Fricção:** Análise de dados de eventos (logs, CSVs) para identificar os principais pontos de abandono (drop-off) em funis de conversão (ex: funil de *sign-up* ou *checkout*) [1].
*   **Síntese de Entrevistas:** Codificação de transcrições por tema e sentimento, resultando em listas classificadas de problemas de usabilidade e oportunidades de design [1].
*   **Análise Quantitativa Básica:** Processamento de dados de pesquisas fechadas (Excel, CSV) para calcular médias, porcentagens e diferenças estatisticamente significativas entre grupos de usuários [1].

**3. Estratégia e Planejamento de UX:**
*   **Geração de Personas:** Criação de personas concisas baseadas em dados demográficos e comportamentais fornecidos [1].
*   **Resumos Executivos:** Transformação de relatórios de pesquisa longos em resumos de uma página para stakeholders de alto nível (C-level), focando em *insights* e ações recomendadas [1].
*   **Geração de Hipóteses:** Criação de hipóteses de design testáveis a partir de pontos de fricção identificados, seguindo formatos estruturados (ex: "Acreditamos que...") [1].

## Pitfalls
**1. Viés e Perguntas Indutoras (Leading Questions):** A IA pode gerar perguntas tendenciosas se não for explicitamente instruída a ser neutra. O treinamento de dados da IA pode introduzir vieses cognitivos, resultando em dados de pesquisa distorcidos [1].
**2. Falta de Nuance e Diversidade:** A IA tende a convergir para a "norma" estatística, o que pode filtrar ou desconsiderar perspectivas únicas e diversas de participantes, especialmente em análises temáticas. A iteração excessiva com a IA pode "despir" a singularidade dos dados qualitativos [1].
**3. Alucinações e Imprecisão:** A IA pode "alucinar" (inventar) fatos, fontes ou dados. É crucial que o pesquisador humano revise e valide todas as saídas, especialmente as referências e a síntese de dados [1].
**4. Dependência Excessiva:** Confiar cegamente na IA para gerar roteiros de pesquisa ou analisar dados sem supervisão humana pode levar a resultados superficiais ou incorretos. A IA é uma ferramenta de processamento poderosa, mas requer supervisão especializada [1].
**5. Preocupações com Privacidade e Conformidade:** O processamento de dados de usuários (transcrições, comentários) por meio de ferramentas de IA levanta questões de privacidade (GDPR, LGPD). É essencial garantir o consentimento do usuário e a anonimização adequada dos dados antes de alimentar a IA [1].
**6. Contexto Insuficiente:** Prompts vagos ou genéricos resultarão em saídas igualmente vagas. A falha em fornecer o contexto (público, produto, objetivo) e as regras (formato, restrições) é um erro comum que anula o benefício da ferramenta [1].

## URL
[https://maze.co/collections/ai/user-research-prompts/](https://maze.co/collections/ai/user-research-prompts/)
