# Instruction Prompting

## Description

**Prompting por Instrução (Instruction Prompting)** é a técnica fundamental e mais comum de engenharia de prompt, baseada em fornecer **instruções explícitas e claras** em linguagem natural (geralmente em frases imperativas ou diretivas) a um Large Language Model (LLM) para que ele execute uma tarefa específica. É o modo de interação padrão com LLMs, permitindo que o modelo realize tarefas sem a necessidade de exemplos (zero-shot) ou modificação de seus parâmetros internos (diferente de Instruction Tuning). A eficácia reside na clareza e especificidade da instrução, que deve detalhar o contexto, o resultado esperado, o formato, o estilo e o comprimento. É crucial usar delimitadores claros (como `###` ou `"""`) para separar a instrução do contexto ou dados de entrada.

## Statistics

**Eficácia:** É o método mais simples e eficaz para interagir com LLMs, especialmente para tarefas de propósito geral.
**Citações:** O conceito é fundamental e amplamente citado em pesquisas de Prompt Engineering. Artigos de 2023 e 2024 (como Longpre et al., 2023) sobre Instruction Tuning e Prompting demonstram a importância da clareza da instrução.
**Trade-offs:** Estudos (como o de código assistido por LLM) indicam que o Prompting por Instrução Direta é mais **flexível** do que outras técnicas, mas pode exigir mais refinamento da instrução para atingir a fidelidade desejada.

## Features

1.  **Diretividade:** Uso de linguagem imperativa ("Resuma", "Crie", "Traduza").
2.  **Clareza e Especificidade:** A instrução deve ser o mais detalhada possível sobre o contexto, o resultado esperado, o formato, o estilo e o comprimento.
3.  **Separação de Contexto:** Uso de delimitadores (como `###` ou `"""`) para separar claramente a instrução do contexto ou dados de entrada.
4.  **Zero-Shot:** Capacidade de obter resultados satisfatórios sem a necessidade de fornecer exemplos de entrada/saída (few-shot).

## Use Cases

1.  **Resumo de Texto:** "Resuma o seguinte artigo em 50 palavras."
2.  **Tradução:** "Traduza a seguinte frase para o português do Brasil."
3.  **Geração de Código:** "Escreva uma função Python para calcular a sequência de Fibonacci."
4.  **Extração de Informações:** "Extraia todos os nomes e datas do texto abaixo."
5.  **Reescrita de Estilo:** "Reescreva o parágrafo a seguir em um tom formal e acadêmico."

## Integration

**Melhores Práticas:**
*   **Posicionamento:** Coloque as instruções no **início** do prompt.
*   **Restrições Positivas:** Diga ao modelo o que **fazer**, em vez de apenas o que **não fazer**.
*   **Formato de Saída Articulado:** Use exemplos ou marcadores para definir o formato de saída.

**Exemplos de Prompt:**
1.  `Instrução: Resuma o texto abaixo em três bullet points. Texto: """[TEXTO AQUI]"""`
2.  `Instrução: Extraia os nomes das empresas e os formate como uma lista separada por vírgulas. Formato Desejado: Nomes de Empresas: <lista_separada_por_virgulas>`
3.  `Instrução: O agente deve diagnosticar o problema e sugerir uma solução, referindo o usuário ao artigo de ajuda www.site.com/faq em vez de pedir informações pessoais.`

## URL

https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-the-openai-api