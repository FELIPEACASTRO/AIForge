# Engenharia de Prompt para Escrita Científica: Regras e Aplicações de LLMs

## Description

A Engenharia de Prompt para Escrita Científica (Scientific Writing Prompts) é um conjunto de técnicas e melhores práticas para a utilização de Large Language Models (LLMs) no processo de pesquisa e redação acadêmica. O foco principal é maximizar os benefícios das LLMs, como a aceleração da escrita e a assistência na codificação, enquanto se minimizam os riscos inerentes, como a alucinação e o plágio. As diretrizes se concentram em estabelecer salvaguardas éticas e metodológicas, além de sugerir usos práticos para otimizar o fluxo de trabalho científico. A abordagem enfatiza a transparência, a verificação factual e a adesão às políticas editoriais das revistas.

## Statistics

O uso de LLMs pode colocar até 300 milhões de empregos em risco globalmente [1]. A popularidade do ChatGPT estabeleceu um recorde para a base de usuários de crescimento mais rápido na história [3]. O artigo principal que estabelece as "Dez Regras Simples" foi publicado no *PLoS Computational Biology* em janeiro de 2024 [2], sendo um recurso fundamental para o consenso acadêmico recente. O estudo destaca o risco de "alucinação" da LLM, citando exemplos onde o modelo forneceu informações factualmente incorretas, mas com referências falsas ou equivocadas [2]. A taxa de citação do artigo principal (14 citações em 2024) indica sua rápida adoção como referência para diretrizes éticas no uso de LLMs na ciência [2].

## Features

**Salvaguardas (Regras 1-5):**
1.  **Adesão às Regras do Periódico:** Consultar e seguir as diretrizes da revista-alvo sobre o uso de LLMs.
2.  **Avaliação de Riscos:** Delinear riscos relevantes (ex: viés, desigualdade de acesso) antes de usar a LLM.
3.  **Prevenção de Plágio:** Evitar o uso de conteúdo gerado pela LLM sem atribuição e garantir que o uso não constitua plágio.
4.  **Confidencialidade:** Não compartilhar dados confidenciais ou resultados preliminares não publicados com a LLM.
5.  **Verificação Factual:** Sempre verificar a veracidade de todo o conteúdo gerado pela LLM por um especialista no assunto.

**Sugestões de Uso (Regras 6-10):**
6.  **Busca de Dados Inclusiva:** Usar LLMs para auxiliar na coleta de "literatura cinzenta" (relatórios de ONGs/governo) para meta-análises.
7.  **Sumarização de Conteúdo:** Gerar resumos concisos de artigos longos ou atas de reunião para otimizar o tempo de leitura.
8.  **Refinamento de Escrita:** Usar LLMs para refinar o inglês escrito (gramática, tom, idiomatismo), especialmente para falantes não nativos.
9.  **Melhoria de Código:** Gerar *snippets* de código, depurar erros e traduzir código entre linguagens de programação.
10. **Início da Escrita:** Superar o bloqueio criativo e a ansiedade da "página em branco" gerando esboços e estruturas de artigos.

## Use Cases

nan

## Integration

**1. Geração de Esboço de Artigo (Regra 10):**
*   **Prompt:** "Atue como um revisor científico sênior. Gere uma estrutura de 4 seções (Introdução, Revisão de Literatura, Metodologia, Resultados e Discussão) para um artigo de pesquisa. O tópico é: [Efeitos da Mudança Climática na Biodiversidade em Ecossistemas Tropicais]. O contexto é [Ecologia] e o tom deve ser [Formal e Acadêmico]. Inclua 3 a 4 subseções para cada seção principal."

**2. Sumarização de Artigo (Regra 7):**
*   **Prompt:** "Quero que você atue como um sumarizador de artigos científicos. Vou fornecer o texto de um artigo. Responda com um título em negrito para cada seção, incluindo: Informações Gerais, Contexto, Questão/Hipótese, Principais Descobertas e Contribuições. O resumo de cada seção deve ser conciso, claro e informativo. [Insira o texto do artigo aqui]."

**3. Depuração de Código (Regra 9):**
*   **Prompt:** "Estou usando o Google Earth Engine e recebi o erro 'Too many concurrent aggregations' com o seguinte código: [Insira o código problemático]. Identifique a causa do erro e sugira uma solução usando a função `ee.List.slice()` para dividir a lista de IDs em blocos menores."

**Melhores Práticas:**
*   **Contextualização:** Sempre defina o papel da LLM (ex: "Atue como um revisor sênior", "Você é um assistente de pesquisa").
*   **Restrições:** Especifique o formato de saída (ex: "Estrutura de 4 seções", "Lista em formato Markdown"), o tom e o público-alvo.
*   **Iteração:** Use o resultado do prompt inicial como base para prompts de acompanhamento (ex: "Expanda a subseção B da Revisão de Literatura").

## URL

https://pmc.ncbi.nlm.nih.gov/articles/PMC10829980/