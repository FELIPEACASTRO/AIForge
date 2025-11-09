# Rubric Design Prompts

## Description
A técnica de **Rubric Design Prompts** (Prompts de Design de Rubrica) refere-se à engenharia de prompts estruturados e detalhados para que Modelos de Linguagem Grande (LLMs) gerem rubricas de avaliação de alta qualidade, claras e personalizadas. O objetivo é automatizar a parte mais desafiadora da criação de rubricas: a formulação de **descritores de qualidade** específicos para cada critério e nível de desempenho. Um prompt eficaz deve incluir: o **papel** da IA (ex: especialista em avaliação), a **tarefa** ou **projeto** a ser avaliado, os **objetivos de aprendizagem** do curso, a **escala de pontuação** desejada (ex: 4 níveis), os **critérios de avaliação** específicos e uma instrução clara para que os descritores se concentrem na **qualidade** e não apenas na quantidade do trabalho. Esta técnica é amplamente utilizada na área de Educação para economizar tempo e garantir a consistência da avaliação.

## Examples
```
**Exemplo 1: Rubrica para Ensaio Argumentativo (Educação)**

> **Papel:** Você é um especialista em avaliação educacional e design de rubricas.
> **Tarefa:** Criar uma rubrica analítica para um ensaio argumentativo de 1500 palavras sobre "O Impacto da IA na Ética Jornalística".
> **Objetivos de Aprendizagem:** O aluno deve demonstrar: 1) Capacidade de formular uma tese clara e defensável; 2) Uso de evidências de fontes confiáveis; 3) Estrutura lógica e coesa; 4) Consciência das implicações éticas.
> **Escala:** 4 níveis: Excelente (4), Bom (3), Satisfatório (2), Insuficiente (1).
> **Critérios:** Tese e Argumento Central, Uso de Evidências e Fontes, Estrutura e Coesão, Análise Ética.
> **Instrução:** Gere a rubrica em formato de tabela Markdown. Para cada critério e nível, crie um descritor que se concentre na **qualidade** da execução e na profundidade da análise, usando linguagem clara e orientada ao aluno.

**Exemplo 2: Rubrica para Revisão de Código (Tecnologia)**

> **Papel:** Atue como um Engenheiro de Software Sênior especializado em qualidade de código Python.
> **Tarefa:** Criar uma rubrica para avaliar a qualidade de um Pull Request (PR) de um desenvolvedor júnior. O PR implementa um novo endpoint de API REST.
> **Critérios:** Legibilidade e Estilo (PEP 8), Eficiência e Performance, Cobertura de Testes Unitários, Documentação (Docstrings e Comentários), Tratamento de Erros.
> **Escala:** 3 níveis: Padrão Sênior (3), Aceitável (2), Requer Refatoração (1).
> **Instrução:** Gere a rubrica em formato de lista aninhada. Os descritores devem ser técnicos e práticos, detalhando o que constitui um código de "Padrão Sênior" em cada critério.

**Exemplo 3: Rubrica para Design de Experiência do Usuário (Design)**

> **Papel:** Você é um Designer de UX/UI com foco em usabilidade e acessibilidade.
> **Tarefa:** Desenvolver uma rubrica para avaliar um protótipo de baixa fidelidade de um aplicativo móvel de gerenciamento financeiro.
> **Objetivos:** Avaliar a navegação intuitiva, a conformidade com as diretrizes de acessibilidade (WCAG 2.1) e a eficácia na resolução do problema do usuário.
> **Critérios:** Usabilidade (Fluxo de Tarefas), Acessibilidade (Contraste e Tamanho da Fonte), Consistência Visual, Resolução do Problema.
> **Escala:** 5 níveis: Excede as Expectativas (5), Atende Plenamente (4), Atende Parcialmente (3), Abaixo do Esperado (2), Não Atende (1).
> **Instrução:** A rubrica deve ser entregue em formato de tabela. Para o critério "Acessibilidade", os descritores devem fazer referência a princípios específicos do WCAG.

**Exemplo 4: Rubrica para Avaliação de Desempenho (Negócios/RH)**

> **Papel:** Consultor de Recursos Humanos especializado em avaliação de desempenho 360 graus.
> **Tarefa:** Criar uma rubrica para avaliar o desempenho trimestral de um Gerente de Projetos.
> **Critérios:** Liderança e Mentoria de Equipe, Gestão de Risco e Orçamento, Comunicação com Stakeholders, Entrega de Resultados (Prazo e Qualidade).
> **Escala:** 4 níveis: Excepcional, Supera as Expectativas, Atende as Expectativas, Necessita Melhoria.
> **Instrução:** Gere uma rubrica concisa. Os descritores devem ser comportamentais e mensuráveis, descrevendo ações observáveis em cada nível de desempenho.

**Exemplo 5: Rubrica para Postagem em Mídia Social (Marketing)**

> **Papel:** Estrategista de Marketing Digital e Copywriter.
> **Tarefa:** Criar uma rubrica para avaliar a eficácia de uma postagem única no Instagram para o lançamento de um novo produto.
> **Critérios:** Engajamento (Taxa de Cliques/Comentários), Clareza da Mensagem (Proposta de Valor), Qualidade Visual (Alinhamento com a Marca), Chamada para Ação (CTA).
> **Escala:** 3 níveis: Alto Impacto, Médio Impacto, Baixo Impacto.
> **Instrução:** Gere a rubrica em formato de tabela. Inclua uma coluna para "Peso" (em %) para cada critério, sendo o Engajamento o mais pesado (40%).
```

## Best Practices
**1. Estrutura Modular:** Divida o prompt em seções claras (Papel, Tarefa, Objetivos, Critérios, Escala, Instruções para Descritores).
**2. Foco na Qualidade:** Instrua explicitamente a IA a gerar descritores que se concentrem na **qualidade** do trabalho (profundidade de compreensão, clareza, precisão) e não apenas na quantidade.
**3. Especificidade é Chave:** Forneça o máximo de detalhes possível sobre a tarefa, os objetivos de aprendizagem e os critérios. Quanto mais específico, mais alinhada será a rubrica.
**4. Linguagem do Público:** Peça à IA para usar uma linguagem apropriada para o público (ex: "linguagem amigável ao aluno" ou "linguagem técnica para pares").
**5. Formato de Saída:** Especifique o formato de saída desejado (ex: "Gerar a rubrica em formato de tabela Markdown").
**6. Revisão Humana:** Sempre revise e ajuste a rubrica gerada pela IA. Ela é uma ferramenta de rascunho, não um produto final.

## Use Cases
**1. Educação e Avaliação:** O caso de uso primário. Professores e instrutores usam para criar rapidamente rubricas para ensaios, projetos, apresentações, exames orais e trabalhos de laboratório, garantindo transparência e consistência na correção.
**2. Desenvolvimento de Software:** Equipes de engenharia usam para criar rubricas de revisão de código (Code Review Rubrics), avaliando critérios como legibilidade, performance, segurança e cobertura de testes.
**3. Design de Produto (UX/UI):** Designers usam para avaliar protótipos, testes de usabilidade e artefatos de design, focando em critérios como usabilidade, acessibilidade e alinhamento com as necessidades do usuário.
**4. Gestão de Desempenho (RH):** Departamentos de Recursos Humanos usam para desenvolver rubricas de avaliação de desempenho de funcionários, definindo expectativas claras para diferentes níveis de senioridade e funções.
**5. Criação de Conteúdo e Marketing:** Profissionais de marketing usam para criar rubricas de qualidade de conteúdo (blog posts, vídeos, posts em mídias sociais), avaliando engajamento, SEO, clareza da mensagem e alinhamento com a marca.

## Pitfalls
**1. Descritores Genéricos:** O erro mais comum é não especificar o suficiente, resultando em descritores vagos como "Bom trabalho" ou "Fez tudo". A IA precisa ser instruída a focar em **qualidade e especificidade**.
**2. Foco na Quantidade:** O prompt falha ao instruir a IA a descrever a **qualidade** do desempenho, resultando em descritores que apenas contam itens (ex: "Incluiu 5 fontes" em vez de "Integrou 5 fontes de forma crítica e pertinente").
**3. Critérios Desalinhados:** Não incluir os **Objetivos de Aprendizagem** ou os requisitos da tarefa no prompt. Isso faz com que a rubrica gerada avalie habilidades que não são o foco do trabalho.
**4. Escala Inadequada:** Usar uma escala de pontuação (ex: 1 a 10) sem definir claramente o que cada ponto significa. A IA precisa de rótulos de qualidade (Exemplar, Proficiente) para criar descritores úteis.
**5. Prompt Único e Longo:** Tentar incluir todas as informações em um único bloco de texto sem formatação. A IA processa melhor prompts estruturados e modulares.

## URL
[https://blog.ctl.gatech.edu/2024/05/01/unlocking-academic-excellence-using-generative-ai-to-create-custom-rubrics/](https://blog.ctl.gatech.edu/2024/05/01/unlocking-academic-excellence-using-generative-ai-to-create-custom-rubrics/)
