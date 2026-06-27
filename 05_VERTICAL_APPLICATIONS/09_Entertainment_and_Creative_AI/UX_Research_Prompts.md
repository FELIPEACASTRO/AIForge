# UX Research Prompts

## Description
**UX Research Prompts** são instruções estruturadas e detalhadas fornecidas a Modelos de Linguagem Grande (LLMs), como ChatGPT ou Gemini, para automatizar, acelerar e aprimorar diversas etapas do processo de Pesquisa de Experiência do Usuário (UX). Em vez de usar a IA apenas para tarefas genéricas, esses prompts são projetados para simular o raciocínio de um pesquisador de UX, auxiliando na geração de planos de pesquisa, roteiros de entrevistas, análise de dados qualitativos (como transcrições), criação de personas, identificação de hipóteses e síntese de descobertas. A adoção de frameworks de prompting específicos (como REFINE, CARE e RACEF) é uma **melhor prática** essencial para garantir que a saída da IA seja relevante, acionável e alinhada aos objetivos de pesquisa. A tendência, observada entre 2023 e 2025, é que a IA se torne uma ferramenta indispensável para aumentar a velocidade e a escala da pesquisa de UX, permitindo que os pesquisadores se concentrem em tarefas de maior valor, como a interpretação e a estratégia.

## Examples
```
**1. Geração de Plano de Pesquisa (Framework REFINE)**
```
**Role:** Pesquisador de UX Sênior.
**Expectation:** Gerar um plano de pesquisa de usabilidade detalhado.
**Format:** Tabela Markdown com colunas: Fase, Objetivo, Método, Duração Estimada.
**Iterate:** Focar na fase de Teste de Usabilidade Remoto.
**Nuance:** O produto é um aplicativo móvel de finanças para a Geração Z.
**Example:** Incluir métricas como Taxa de Sucesso da Tarefa e Tempo na Tarefa.

**Prompt:** "Assuma o papel de um Pesquisador de UX Sênior. Gere um plano de pesquisa de usabilidade detalhado para um aplicativo móvel de finanças focado na Geração Z. O plano deve ser apresentado em uma Tabela Markdown com as colunas: Fase, Objetivo, Método e Duração Estimada. Dê foco especial à fase de Teste de Usabilidade Remoto e inclua métricas como Taxa de Sucesso da Tarefa e Tempo na Tarefa."
```

**2. Análise de Transcrição Qualitativa**
```
**Role:** Analista de Dados Qualitativos.
**Context:** [INSERIR TRANSCRIÇÃO DA ENTREVISTA AQUI].
**Ask:** Identificar os 5 principais pontos de dor (pain points) e as 3 principais necessidades não atendidas (unmet needs) do usuário.
**Rules:** A saída deve ser uma lista numerada, citando trechos da transcrição para apoiar cada ponto.
**Example:** O formato deve ser: "Ponto de Dor 1: [Descrição] - Citação: '...'"

**Prompt:** "Assuma o papel de um Analista de Dados Qualitativos. Analise a transcrição da entrevista fornecida abaixo. Identifique os 5 principais pontos de dor e as 3 principais necessidades não atendidas. A saída deve ser uma lista numerada, citando trechos exatos da transcrição para apoiar cada ponto. [INSERIR TRANSCRIÇÃO DA ENTREVISTA AQUI]"
```

**3. Criação de Roteiro de Entrevista (Framework CARE)**
```
**Context:** Estamos desenvolvendo um novo recurso de "Lista de Desejos Colaborativa" para um e-commerce de decoração.
**Ask:** Crie um roteiro de entrevista semi-estruturado para 60 minutos.
**Rules:** O roteiro deve ter 5 seções (Introdução, Uso Atual, Necessidades do Recurso, Teste de Conceito, Encerramento). As perguntas devem ser abertas e não indutoras.
**Examples:** Evitar perguntas como 'Você gostou do recurso?'. Preferir 'O que você faria com este recurso?'

**Prompt:** "Crie um roteiro de entrevista semi-estruturado de 60 minutos para um estudo sobre um novo recurso de 'Lista de Desejos Colaborativa' em um e-commerce de decoração. O roteiro deve ser dividido em 5 seções: Introdução, Uso Atual, Necessidades do Recurso, Teste de Conceito e Encerramento. As perguntas devem ser abertas e não indutoras. Evite perguntas de 'sim/não'."
```

**4. Geração de Hipóteses de Usabilidade**
```
**Role:** Especialista em Heurísticas de Nielsen.
**Context:** O mapa de calor (heatmap) mostra que 70% dos usuários abandonam a página de checkout na etapa de 'Informações de Envio'.
**Ask:** Gerar 5 hipóteses de usabilidade testáveis para explicar o abandono.
**Rules:** Cada hipótese deve seguir o formato: 'Acreditamos que [AÇÃO DO USUÁRIO] porque [PROBLEMA DE USABILIDADE], o que resultará em [MÉTRICA DE SUCESSO]'.

**Prompt:** "Assuma o papel de um Especialista em Heurísticas de Nielsen. O mapa de calor indica que 70% dos usuários abandonam a página de checkout na etapa de 'Informações de Envio'. Gere 5 hipóteses de usabilidade testáveis para explicar esse abandono. Cada hipótese deve seguir o formato: 'Acreditamos que [AÇÃO DO USUÁRIO] porque [PROBLEMA DE USABILIDADE], o que resultará em [MÉTRICA DE SUCESSO]'."
```

**5. Síntese de Descobertas para Stakeholders**
```
**Role:** Comunicador de Pesquisa.
**Context:** [INSERIR RESUMO DAS DESCOBERTAS AQUI - ex: 10 entrevistas, 3 testes de usabilidade].
**Ask:** Criar um resumo executivo de 5 pontos para a liderança.
**Rules:** O resumo deve ser focado em implicações de negócios e recomendações de design de alto nível. Usar linguagem clara e direta.

**Prompt:** "Com base nas descobertas de pesquisa fornecidas abaixo, crie um resumo executivo de 5 pontos para a liderança. O resumo deve focar nas implicações de negócios e em recomendações de design de alto nível. Use linguagem clara e direta. [INSERIR RESUMO DAS DESCOBERTAS AQUI]"
```
```

## Best Practices
**1. Adote um Framework de Prompting:** Utilize estruturas como **REFINE** (Role, Expectation, Format, Iterate, Nuance, Example), **CARE** (Context, Ask, Rules, Examples) ou **RACEF** (Rephrase, Append, Clarify, Examples, Focus) para garantir que o LLM receba todas as informações necessárias para uma resposta de alta qualidade.
**2. Defina o Papel (Role) Claramente:** Comece o prompt instruindo a IA a assumir a persona de um "Pesquisador de UX Sênior", "Analista de Dados de UX" ou "Especialista em Usabilidade". Isso alinha o tom e o foco da resposta.
**3. Forneça Contexto e Dados de Entrada:** A IA é "cega" sem contexto. Inclua transcrições de entrevistas, dados de pesquisa, personas, ou o problema de design específico que está sendo abordado.
**4. Especifique o Formato de Saída:** Peça o resultado em um formato estruturado (ex: "Tabela Markdown", "Lista numerada", "Resumo de 5 pontos") para facilitar a análise e a integração no seu fluxo de trabalho.
**5. Itere e Refine:** O primeiro resultado raramente é o final. Use prompts de acompanhamento (Iterate/Clarify) para refinar, adicionar nuances, remover seções irrelevantes ou aprofundar em um ponto específico.
**6. Use Exemplos (Examples) e Regras (Rules):** Incluir exemplos do que você quer (ou não quer) e regras claras (ex: "Perguntas não podem ser indutoras", "Foco apenas em problemas de usabilidade") melhora drasticamente a precisão.

## Use Cases
nan

## Pitfalls
**1. Confiar Cegamente na Saída da IA:** A IA pode gerar conteúdo plausível, mas factualmente incorreto ou enviesado (o chamado "hallucination"). O pesquisador de UX **deve** sempre revisar, validar e aplicar seu julgamento profissional à saída da IA.
**2. Prompts Genéricos ou Vagos:** Prompts como "Me ajude com pesquisa de UX" resultam em respostas superficiais. A falta de contexto, regras e formato específico é o erro mais comum.
**3. Ignorar a Ética e a Privacidade:** Usar dados sensíveis de usuários (PII - Personally Identifiable Information) em prompts de IA sem anonimização e consentimento adequados é uma violação ética e legal.
**4. Falha em Iterar:** Tratar a IA como um oráculo de resposta única. A força do prompting de UX reside na **iteração** e no refinamento contínuo da resposta.
**5. Viés de Confirmação:** Pedir à IA para confirmar uma hipótese pré-existente em vez de pedir uma análise neutra. Isso pode reforçar vieses em vez de desafiá-los.
**6. Não Definir o Papel (Role):** Sem um papel claro (ex: "Atue como um crítico de usabilidade"), a IA pode responder com um tom ou foco inadequado para a tarefa de pesquisa.

## URL
[https://maze.co/collections/ai/user-research-prompts/](https://maze.co/collections/ai/user-research-prompts/)
