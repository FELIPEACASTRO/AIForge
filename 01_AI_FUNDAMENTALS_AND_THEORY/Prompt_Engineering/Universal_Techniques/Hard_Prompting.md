# Hard Prompting

## Description
Hard Prompting, ou 'Prompting Explícito', refere-se à técnica fundamental de engenharia de prompt onde as instruções são fornecidas ao modelo de linguagem (LLM) como texto explícito, legível por humanos e em linguagem natural. Diferentemente do Soft Prompting, que utiliza vetores de embedding otimizados e inacessíveis ao usuário, o Hard Prompting é totalmente transparente e editável. A eficácia desta técnica depende diretamente da clareza, especificidade e criatividade do usuário, sendo a base para todas as técnicas avançadas de prompting, como Chain-of-Thought (CoT), Few-Shot Prompting e Role Prompting. É a abordagem preferencial para prototipagem rápida, tarefas de propósito geral e cenários onde a interpretabilidade e o controle manual sobre a entrada são cruciais.

## Examples
```
1. **Role Prompting (Definição de Papel):**

   `Aja como um analista de marketing sênior. Sua tarefa é analisar o seguinte relatório de vendas trimestrais e identificar as três principais oportunidades de crescimento para o próximo trimestre. Apresente sua análise em formato de lista com marcadores.`

2. **Few-Shot Prompting (Aprendizado com Exemplos):**

   `Classifique os seguintes sentimentos como Positivo, Negativo ou Neutro. Aqui estão três exemplos:

   Entrada: 'O serviço foi lento, mas a comida estava ótima.'
   Saída: Neutro

   Entrada: 'Experiência terrível, nunca mais voltarei.'
   Saída: Negativo

   Entrada: 'Adorei o novo design do site, muito intuitivo.'
   Saída: Positivo

   Entrada: 'A entrega atrasou 10 minutos, mas o produto veio intacto.'
   Saída: ?`

3. **Chain-of-Thought (Cadeia de Pensamento):**

   `O custo de produção de um widget é R$ 50. O preço de venda é R$ 80. Se uma empresa vende 1000 widgets, qual é o lucro total? Pense passo a passo antes de dar a resposta final.`

4. **Restrição de Formato (JSON):**

   `Extraia o nome, o cargo e o e-mail de contato do texto abaixo. Retorne o resultado estritamente no formato JSON, seguindo o esquema {\"nome\": \"\", \"cargo\": \"\", \"email\": \"\"}.`

5. **Geração de Código:**

   `Escreva uma função em Python chamada 'fibonacci' que receba um número inteiro 'n' e retorne o n-ésimo número da sequência de Fibonacci. Inclua comentários explicando a lógica.`

6. **Instrução de Edição e Revisão:**

   `Revise o parágrafo a seguir para clareza, concisão e tom profissional. Corrija quaisquer erros gramaticais e sugira uma frase de abertura mais forte. [Parágrafo a ser revisado]`

7. **Prompt Criativo com Restrições:**

   `Escreva um microconto de ficção científica (máximo 100 palavras) sobre um robô que descobre a chuva pela primeira vez. O conto deve ter um tom melancólico e terminar com a palavra 'silêncio'.`

8. **Instrução de Resumo e Análise:**

   `Leia o artigo abaixo e forneça um resumo de 5 pontos-chave. Em seguida, analise o público-alvo provável do artigo e o principal argumento do autor.`

9. **Prompt de Tradução com Contexto:**

   `Traduza a seguinte frase do Português para o Inglês, mantendo um tom formal e de negócios: 'A implementação do novo protocolo de segurança é imperativa para a conformidade regulatória.'`

10. **Instrução de Categorização:**

    `Classifique o seguinte livro em uma das categorias: Ficção, Não-Ficção, Biografia, Poesia. Justifique sua escolha em uma frase. [Título e breve sinopse do livro]`
```

## Best Practices
O Hard Prompting é a base da interação com LLMs. As melhores práticas envolvem:

*   **Seja Explícito e Específico:** Defina claramente a tarefa, o formato de saída e quaisquer restrições. Evite ambiguidades.
*   **Defina um Papel (Role Prompting):** Atribuir uma persona (ex: 'Aja como um historiador') melhora a qualidade e o foco da resposta.
*   **Use Exemplos (Few-Shot):** Para tarefas complexas ou que exigem um formato específico, fornecer 1 a 3 exemplos de entrada/saída aumenta drasticamente a precisão.
*   **Instrua o Raciocínio (CoT):** Peça ao modelo para 'pensar passo a passo' ou 'explicar seu raciocínio' antes de dar a resposta final, o que melhora a lógica e a precisão.
*   **Isole a Tarefa:** Coloque as instruções principais e o contexto em seções separadas ou use delimitadores (como aspas triplas) para evitar confusão.
*   **Itere e Refine:** Otimize o prompt por tentativa e erro, ajustando a fraseologia até obter o resultado desejado.

## Use Cases
O Hard Prompting é aplicável a praticamente todos os casos de uso de LLMs, sendo ideal para:

*   **Geração de Conteúdo:** Criação de artigos, e-mails, posts de blog e roteiros.
*   **Resumo e Extração de Informação:** Condensar documentos longos e extrair dados estruturados.
*   **Tradução e Localização:** Tradução de textos com requisitos de tom e contexto específicos.
*   **Geração de Código:** Escrever funções, scripts e snippets de código para tarefas específicas.
*   **Resolução de Problemas Lógicos:** Utilizando técnicas como Chain-of-Thought para resolver problemas matemáticos ou de raciocínio.
*   **Prototipagem Rápida:** Testar rapidamente ideias e funcionalidades sem a necessidade de fine-tuning do modelo.

## Pitfalls
Embora seja versátil, o Hard Prompting apresenta armadilhas comuns:

*   **Dependência da Habilidade Humana:** A qualidade da saída é limitada pela clareza e criatividade do prompt humano. Prompts mal escritos resultam em saídas ruins ('Garbage In, Garbage Out').
*   **Ineficiência para Tarefas de Alta Precisão:** Para tarefas altamente especializadas (como análise de sentimento sutil ou detecção de anomalias), o Hard Prompting pode ser menos preciso do que o Soft Prompting otimizado.
*   **Prompts Excessivamente Longos:** Tentar incluir muito contexto ou muitas regras em um único prompt pode levar o modelo a se confundir ou a ignorar partes das instruções.
*   **Ambiguidade:** O uso de linguagem vaga ou termos com múltiplos significados pode levar a interpretações incorretas pelo modelo.
*   **Custo de Iteração:** Otimizar prompts complexos requer muitas tentativas e erros manuais, o que pode ser demorado e caro em termos de tokens.

## URL
[https://futureagi.com/blogs/hard-prompt-vs-soft-prompt-2025](https://futureagi.com/blogs/hard-prompt-vs-soft-prompt-2025)
