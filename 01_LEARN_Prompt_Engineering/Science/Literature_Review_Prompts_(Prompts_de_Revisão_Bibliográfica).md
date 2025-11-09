# Literature Review Prompts (Prompts de Revisão Bibliográfica)

## Description
**Prompts de Revisão Bibliográfica (Literature Review Prompts)** são instruções especializadas projetadas para alavancar Modelos de Linguagem Grande (LLMs), como o ChatGPT, Gemini ou Claude, na execução de tarefas complexas e multifacetadas inerentes ao processo de revisão sistemática ou narrativa da literatura acadêmica. Em vez de apenas solicitar informações factuais, esses prompts são estruturados para guiar a IA a realizar funções de pesquisa de nível superior, como **síntese**, **análise crítica**, **identificação de lacunas** e **extração de dados estruturados** de textos científicos. A eficácia desses prompts reside na sua capacidade de decompor o processo de revisão (que tradicionalmente exige meses de trabalho humano) em etapas gerenciáveis e automatizadas, resultando em saídas mais precisas, relevantes e formatadas para uso acadêmico [1] [2]. O uso de técnicas avançadas como **Chain-of-Thought (CoT)** e a especificação de formatos de saída (ex: tabelas Markdown) são cruciais para maximizar a utilidade da IA neste contexto [3].

## Examples
```
**Exemplo 1: Síntese de Metodologias (Synthesis of Methodologies)**

> **Prompt:** "Atue como um Analista de Pesquisa Sênior. Analise os seguintes resumos de artigos [INSERIR RESUMOS AQUI]. Seu objetivo é sintetizar as metodologias de pesquisa primárias utilizadas em cada estudo. Crie uma tabela Markdown com as colunas: 'Estudo', 'Tipo de Metodologia (Qualitativa/Quantitativa/Mista)', 'Tamanho da Amostra/Corpus' e 'Principal Instrumento de Coleta de Dados'. Mantenha a resposta estritamente na tabela."

**Exemplo 2: Identificação de Lacunas de Pesquisa (Identifying Research Gaps)**

> **Prompt:** "Com base na seguinte revisão de literatura [INSERIR TEXTO DA REVISÃO AQUI], identifique as três lacunas de pesquisa mais significativas. Para cada lacuna, forneça uma justificativa concisa (máximo 50 palavras) e sugira uma pergunta de pesquisa futura que aborde essa lacuna. Apresente o resultado em uma lista numerada."

**Exemplo 3: Análise Crítica e Comparação (Critical Analysis and Comparison)**

> **Prompt:** "Compare e contraste as conclusões dos estudos A, B e C sobre o impacto da [TÓPICO ESPECÍFICO]. Concentre-se em identificar pontos de concordância e discordância. Use o formato de parágrafo, com um parágrafo introdutório, um para concordâncias, um para discordâncias e um parágrafo de síntese. Use citações no estilo APA (ex: (Autor, Ano)) para referenciar os estudos."

**Exemplo 4: Extração de Dados Estruturados (Structured Data Extraction)**

> **Prompt:** "Extraia os seguintes dados dos artigos fornecidos [INSERIR TEXTOS AQUI]: Ano de Publicação, País de Origem do Autor Principal, e o principal 'Achado Chave' (Key Finding). Formate a saída como um objeto JSON, onde a chave é o título do artigo e o valor é um objeto contendo os três campos solicitados."

**Exemplo 5: Geração de Introdução de Revisão (Generating Review Introduction)**

> **Prompt:** "Escreva o parágrafo introdutório para uma revisão de literatura sobre '[TEMA ESPECÍFICO]', com foco na evolução do tema entre 2015 e 2025. O parágrafo deve: 1) Declarar a importância do tema; 2) Mencionar a crescente complexidade da pesquisa recente; 3) Apresentar a tese da sua revisão (ex: 'Esta revisão busca mapear as principais tendências metodológicas'). Mantenha o tom acadêmico e formal."

**Exemplo 6: Refinando a Busca (Refining the Search)**

> **Prompt:** "Eu usei as palavras-chave 'aprendizagem de máquina' E 'saúde mental' E 'adolescentes'. Sugira 5 combinações de palavras-chave alternativas (incluindo sinônimos e termos relacionados) que eu deveria usar em um banco de dados acadêmico (como Scopus ou Web of Science) para garantir uma cobertura mais abrangente da literatura. Justifique brevemente cada sugestão."

**Exemplo 7: Identificação de Tendências (Identifying Trends)**

> **Prompt:** "Analise os seguintes títulos e resumos [INSERIR LISTA AQUI]. Qual é a principal tendência metodológica ou teórica emergente que você observa? Forneça uma breve análise (máximo 150 palavras) e cite três artigos que melhor exemplificam essa tendência."
```

## Best Practices
**1. Estrutura e Clareza (Structure and Clarity):**
*   **Defina o Papel (Define the Role):** Comece o prompt instruindo a IA a assumir o papel de um "Revisor de Literatura Sênior", "Pesquisador Acadêmico" ou "Analista de Dados".
*   **Especifique o Objetivo (Specify the Goal):** Seja explícito sobre o que você precisa: "Identificar lacunas de pesquisa", "Sintetizar metodologias", "Comparar resultados de estudos".
*   **Contexto e Escopo (Context and Scope):** Forneça o máximo de contexto possível, incluindo o tópico exato, o período de tempo (ex: 2020-2025), e as palavras-chave principais.

**2. Formato e Restrições (Format and Constraints):**
*   **Exija Formato Estruturado (Demand Structured Format):** Peça a saída em formatos específicos como tabelas Markdown, listas numeradas, ou JSON, para facilitar a análise.
*   **Use CoT (Chain-of-Thought):** Para tarefas complexas (como identificar conflitos ou tendências), peça à IA para "Pensar Passo a Passo" (Chain-of-Thought) antes de dar a resposta final.
*   **Limite a Extensão (Limit Length):** Use frases como "Resuma em 500 palavras" ou "Forneça 3 pontos principais" para manter o foco e evitar divagações.

**3. Iteração e Verificação (Iteration and Verification):**
*   **Entrada de Dados (Data Input):** Sempre que possível, forneça o texto-fonte (resumos, trechos de artigos) diretamente no prompt, em vez de depender do conhecimento interno da IA.
*   **Verificação Cruzada (Cross-Verification):** Use a IA para gerar a análise, mas **sempre** verifique as referências e os fatos citados em fontes primárias. A IA é uma assistente, não a fonte final de verdade acadêmica.
*   **Refinamento (Refinement):** Use prompts de acompanhamento (follow-up) para aprofundar a análise (ex: "Agora, expanda o ponto 3, focando nas implicações éticas").

## Use Cases
**1. Pesquisa Acadêmica e TCCs (Academic Research and Theses):**
*   **Função:** Acelerar a fase inicial de coleta e organização de dados para teses, dissertações e artigos científicos.
*   **Exemplo:** Usar prompts para extrair automaticamente a população, o método estatístico e os resultados-chave de dezenas de resumos de artigos, transformando-os em uma planilha estruturada para análise.

**2. Desenvolvimento de Políticas e Relatórios (Policy Development and Reports):**
*   **Função:** Sintetizar rapidamente o estado da arte sobre um tópico regulatório ou social para informar a tomada de decisão.
*   **Exemplo:** Um analista governamental usa um prompt para resumir as "melhores práticas" e "desafios" de políticas de energia renovável em cinco países diferentes, gerando um relatório comparativo conciso.

**3. Inovação e Desenvolvimento de Produtos (Innovation and Product Development):**
*   **Função:** Identificar lacunas de mercado ou tecnologias emergentes que ainda não foram totalmente exploradas pela concorrência.
*   **Exemplo:** Uma equipe de P&D usa prompts para analisar patentes e artigos recentes, buscando "tecnologias subutilizadas" ou "problemas não resolvidos" em um nicho específico, orientando o desenvolvimento de um novo produto.

**4. Educação e Aprendizagem (Education and Learning):**
*   **Função:** Ajudar estudantes a compreender rapidamente o panorama de um campo de estudo ou a praticar a análise crítica de textos.
*   **Exemplo:** Um professor pede à IA para gerar um "mapa conceitual" ou uma "árvore de sub-tópicos" a partir de um artigo seminal, facilitando a compreensão dos alunos sobre a estrutura da pesquisa.

**5. Jornalismo de Dados e Investigativo (Data and Investigative Journalism):**
*   **Função:** Analisar rapidamente grandes volumes de documentos (ex: relatórios governamentais, documentos vazados) para identificar padrões, contradições ou narrativas principais.
*   **Exemplo:** Um jornalista usa um prompt para extrair todos os "conflitos de interesse" mencionados em um conjunto de relatórios anuais de uma corporação, organizando os dados para uma matéria investigativa.

## Pitfalls
**1. Alucinações e Falsas Citações (Hallucinations and False Citations):**
*   **Erro:** A IA pode inventar artigos, autores, datas ou conclusões que parecem plausíveis, mas são totalmente falsos.
*   **Solução:** **NUNCA** confie cegamente nas referências geradas. Use a IA apenas para processar textos que você mesmo forneceu ou para gerar ideias de busca, mas sempre verifique as fontes primárias.

**2. Dependência do Conhecimento Interno (Over-reliance on Internal Knowledge):**
*   **Erro:** Pedir à IA para "revisar a literatura sobre X" sem fornecer os artigos. A IA usará seu *corpus* de treinamento, que pode estar desatualizado ou ser enviesado.
*   **Solução:** Use prompts de revisão bibliográfica principalmente para **processar e analisar** o texto que você fornece (resumos, artigos completos, notas de pesquisa), e não para buscar a literatura.

**3. Falta de Contexto e Escopo (Lack of Context and Scope):**
*   **Erro:** Prompts vagos como "Me ajude com minha revisão de literatura". A IA não saberá o foco, o público-alvo ou o tipo de análise necessária.
*   **Solução:** Inclua sempre o **papel** da IA, o **tópico exato**, o **formato de saída** e as **restrições** (ex: número de palavras, estilo de citação).

**4. Viés de Confirmação (Confirmation Bias):**
*   **Erro:** A IA pode tender a confirmar suas hipóteses pré-existentes, ignorando ou minimizando estudos que as contradizem, especialmente se o prompt for formulado de forma tendenciosa.
*   **Solução:** Peça explicitamente à IA para **identificar conflitos**, **pontos de vista opostos** ou **limitações** nos estudos. Use prompts neutros e críticos.

**5. Sobrecarga de Informação (Information Overload):**
*   **Erro:** Fornecer um volume excessivo de texto de uma só vez, excedendo o limite de *tokens* da IA ou diluindo o foco.
*   **Solução:** Divida a revisão em tarefas menores e iterativas (ex: "Analise os primeiros 10 resumos", depois "Analise os próximos 10").

## URL
[https://www.nature.com/articles/s41598-025-99423-9](https://www.nature.com/articles/s41598-025-99423-9)
