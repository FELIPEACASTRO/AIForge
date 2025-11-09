# Prompts de Escrita Não-Ficcional (Non-Fiction Writing Prompts)

## Description
**Prompts de Escrita Não-Ficcional** (Non-Fiction Writing Prompts) são instruções estruturadas e detalhadas fornecidas a Modelos de Linguagem Grande (LLMs) para auxiliar na criação de conteúdo factual, informativo e baseado em evidências. Ao contrário da escrita de ficção, o foco principal é a **precisão**, a **credibilidade** e a **estrutura lógica** do argumento ou da informação apresentada [1].

A engenharia de prompt para não-ficção é uma técnica especializada que transforma a IA de um mero gerador de texto em um **assistente de pesquisa, editor e co-autor**. Os prompts são projetados para segmentar o processo de escrita não-ficcional em tarefas gerenciáveis, como:
1.  **Pesquisa e Coleta de Dados**: Resumir relatórios, extrair fatos e identificar tendências.
2.  **Estruturação**: Criar esboços de capítulos, títulos e subtítulos.
3.  **Geração de Conteúdo**: Escrever rascunhos de seções, introduções e conclusões.
4.  **Revisão e Edição**: Verificar a gramática, o tom, a clareza e a conformidade com guias de estilo (ex: APA, ABNT) [2].

A eficácia de um prompt de não-ficção reside na sua capacidade de impor **restrições** e fornecer **contexto** suficiente para que a IA produza uma saída que seja factualmente correta e estilisticamente apropriada para o público-alvo. O uso de IA na escrita não-ficcional, especialmente desde 2023, tem se concentrado em aumentar a **produtividade** e a **qualidade da pesquisa**, permitindo que autores se concentrem na análise e na voz autêntica [3].

## Examples
```
**Exemplos de Prompts (Prompt Examples)**:

1.  **Prompt de Esboço de Livro (Book Outline Prompt)**:
    ```
    Aja como um editor de não-ficção especializado em autoajuda. Crie um esboço detalhado de 5 capítulos para um livro intitulado "A Arte da Produtividade Profunda". O público-alvo são profissionais de tecnologia sobrecarregados. Inclua um título de capítulo, 3 subtítulos principais por capítulo e uma breve descrição do conteúdo de cada seção. O tom deve ser motivacional, mas baseado em ciência.
    ```
2.  **Prompt de Resumo de Pesquisa (Research Summary Prompt)**:
    ```
    Você é um analista de dados. Analise o relatório anexo (URL: [link para o PDF]) sobre "Tendências de Trabalho Remoto no Q2 de 2024". Extraia e resuma os 3 principais insights sobre a produtividade de equipes híbridas. O resumo deve ter no máximo 200 palavras e ser formatado em três parágrafos concisos.
    ```
3.  **Prompt de Geração de Introdução (Introduction Generation Prompt)**:
    ```
    Escreva a introdução para um artigo de blog de 1500 palavras sobre "Os Riscos da IA Generativa na Educação". A introdução deve: 1) Capturar a atenção do leitor com uma estatística chocante (invente uma estatística plausível); 2) Apresentar a tese do artigo (a IA é uma ferramenta poderosa, mas exige novas políticas de integridade); 3) Ter um tom jornalístico e urgente.
    ```
4.  **Prompt de Revisão de Estilo (Style Revision Prompt)**:
    ```
    Revise o parágrafo a seguir para clareza, concisão e conformidade com o Guia de Estilo de Chicago. Remova jargões desnecessários e substitua todas as frases passivas por ativas. O público é acadêmico.

    [Insira o parágrafo a ser revisado aqui]
    ```
5.  **Prompt de Geração de Títulos (Headline Generation Prompt)**:
    ```
    Gere 10 títulos de artigos de blog otimizados para SEO sobre o tema "Sustentabilidade em Pequenas Empresas". Os títulos devem ser atraentes, incluir números (ex: 5, 7, 10) e ter um comprimento máximo de 60 caracteres.
    ```
6.  **Prompt de Criação de Estudo de Caso (Case Study Creation Prompt)**:
    ```
    Crie um estudo de caso de sucesso para uma empresa fictícia de SaaS (Software as a Service) chamada 'DataFlow'. O problema era a baixa retenção de clientes (Churn Rate de 15%). A solução foi a implementação de um novo recurso de onboarding baseado em IA. Descreva o 'Problema', a 'Solução' e os 'Resultados' (Retenção de 95% em 6 meses). O tom deve ser profissional e focado em resultados.
    ```
7.  **Prompt de Fato e Verificação (Fact and Verification Prompt)**:
    ```
    Você é um verificador de fatos. Verifique a precisão da seguinte afirmação: "A energia solar representa 5% da matriz energética global em 2023." Se a afirmação estiver incorreta, forneça o dado correto e a fonte mais recente (URL) que o comprove.
    ```
8.  **Prompt de Expansão de Tópico (Topic Expansion Prompt)**:
    ```
    Expanda o seguinte ponto de tópico em um parágrafo de 150 palavras para um manual técnico. O tom deve ser neutro e informativo.

    Ponto de Tópico: "A importância da criptografia de ponta a ponta em comunicações sensíveis."
    ```
```

## Best Practices
**Melhores Práticas (Best Practices)**:

1.  **Contextualização Detalhada (Detailed Contextualization)**: Sempre inclua o máximo de contexto possível. Especifique o **público-alvo**, o **formato de saída** (artigo, resumo, esboço de livro), o **tom de voz** (acadêmico, conversacional, jornalístico) e a **fonte de dados** (seja um documento específico ou a necessidade de pesquisa na web).
2.  **Definição de Papel (Role Definition)**: Comece o prompt definindo o papel da IA (ex: "Você é um pesquisador sênior em tendências de mercado", "Você é um editor de estilo APA"). Isso alinha a resposta da IA com a perspectiva e o conhecimento necessários.
3.  **Iteração e Refinamento (Iteration and Refinement)**: Use prompts de não-ficção em um processo iterativo. Comece com um prompt de **esboço** (outline), depois um prompt de **rascunho de seção** (section draft), e finalize com um prompt de **revisão/edição** (revision/editing).
4.  **Inclusão de Restrições (Inclusion of Constraints)**: Para garantir a precisão e a originalidade, inclua restrições como "Cite todas as fontes no formato APA" ou "Garanta que o conteúdo seja 100% original e não uma paráfrase de artigos existentes".
5.  **Uso de Dados Externos (External Data Usage)**: Para escrita não-ficcional de alta qualidade, a IA deve ser instruída a analisar dados específicos (documentos, relatórios, URLs) em vez de apenas usar seu conhecimento geral.

**Dicas Adicionais (Additional Tips)**:

*   **Prompt de Voz (Voice Prompt)**: Forneça um pequeno exemplo de sua escrita para que a IA possa replicar seu estilo autêntico.
*   **Prompt de Estrutura (Structure Prompt)**: Peça à IA para gerar uma estrutura de tópicos antes de escrever o conteúdo.

## Use Cases
**Casos de Uso (Use Cases)**:

| Caso de Uso (Use Case) | Descrição (Description) | Benefício (Benefit) |
| :--- | :--- | :--- |
| **Esboço de Livros e Manuais** | Geração de estruturas de capítulos e seções para obras longas de não-ficção (ex: guias técnicos, biografias, livros de autoajuda). | **Aceleração da Estruturação**: Reduz o tempo de planejamento de meses para horas. |
| **Resumos Executivos e Relatórios** | Condensação de documentos longos (ex: relatórios financeiros, artigos científicos, transcrições de reuniões) em resumos concisos e focados. | **Eficiência na Pesquisa**: Permite a rápida digestão de grandes volumes de dados. |
| **Criação de Conteúdo de Marketing** | Geração de artigos de blog, white papers, estudos de caso e e-books baseados em dados para estratégias de marketing de conteúdo. | **Geração de Leads**: Produz conteúdo de autoridade que atrai e educa o público. |
| **Jornalismo e Verificação de Fatos** | Auxílio na pesquisa de antecedentes, verificação de dados e geração de rascunhos de artigos jornalísticos baseados em fontes fornecidas. | **Aumento da Precisão**: Suporta a credibilidade do conteúdo com referências rápidas. |
| **Documentação Técnica** | Criação de manuais de usuário, FAQs e documentação de API com clareza e consistência, seguindo guias de estilo rigorosos. | **Consistência e Clareza**: Garante que a documentação seja acessível e precisa. |

## Pitfalls
**Erros Comuns (Common Pitfalls)**:

1.  **Alucinação Factual (Factual Hallucination)**: O erro mais crítico na escrita não-ficcional com IA. A IA pode gerar informações factualmente incorretas ou inventar citações e fontes. **Mitigação**: Sempre inclua uma etapa de verificação de fatos e instrua a IA a citar fontes verificáveis.
2.  **Generalização Excessiva (Over-Generalization)**: A IA pode produzir conteúdo genérico e superficial se o prompt for muito amplo. **Mitigação**: Seja extremamente específico sobre o tópico, o ângulo e os dados a serem usados.
3.  **Plágio e Paráfrase (Plagiarism and Paraphrasing)**: Sem instruções claras, a IA pode parafrasear conteúdo existente, resultando em problemas de originalidade. **Mitigação**: Inclua a restrição "Gere conteúdo 100% original e cite todas as fontes de dados externos".
4.  **Tom Inconsistente (Inconsistent Tone)**: O tom de voz pode mudar entre as seções geradas por diferentes prompts. **Mitigação**: Defina o tom de voz (ex: acadêmico, informal, autoritário) no início de cada sessão de prompt ou forneça um "prompt de voz" com um exemplo de sua escrita.
5.  **Dependência de Conhecimento Interno (Over-reliance on Internal Knowledge)**: Confiar apenas no conhecimento interno do LLM, que pode estar desatualizado, em vez de dados externos e recentes. **Mitigação**: Sempre que possível, forneça à IA documentos, URLs ou dados específicos para análise.

## URL
[https://clickup.com/p/ai/prompts/non-fiction-writing](https://clickup.com/p/ai/prompts/non-fiction-writing)
