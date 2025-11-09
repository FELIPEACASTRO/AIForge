# Prompts de Chamada para Ação (Call-to-Action Prompts)

## Description
A técnica de **Prompts de Chamada para Ação (CTA Prompts)** em Engenharia de Prompt consiste em incluir uma instrução final, clara e concisa, que orienta o Modelo de Linguagem Grande (LLM) sobre a **ação específica** que deve ser executada após o processamento de todo o contexto e as instruções anteriores. Diferentemente dos CTAs de marketing, que buscam a ação do usuário, o CTA Prompt busca a **ação do próprio LLM**, servindo como um comando de execução ou um gatilho de finalização. Essa técnica é crucial para garantir que o LLM não apenas compreenda a tarefa, mas também a execute no formato e escopo desejados, minimizando a divagação ou a produção de texto introdutório desnecessário. É a parte do prompt que diz ao modelo: "Agora, faça isso." Sua eficácia reside em sua capacidade de focar a atenção do modelo no resultado final desejado, especialmente em prompts longos e complexos. O CTA Prompt atua como o ponto focal que transforma a descrição da tarefa em um comando de execução.

## Examples
```
1. **Geração de Código:**
```
[CONTEXTO: Descrição da funcionalidade e requisitos técnicos...]
[FORMATO: O código deve ser em Python, seguindo o PEP 8.]
### AÇÃO FINAL ###
Gere o código completo da função `calculate_fibonacci(n)` e inclua um exemplo de uso.
```

2. **Resumo Executivo:**
```
[CONTEXTO: O texto a seguir é um relatório de 5.000 palavras sobre tendências de mercado.]
[RESTRICÃO: O resumo deve ter no máximo 250 palavras e focar apenas nas implicações financeiras.]
### AÇÃO FINAL ###
Resuma o relatório em um parágrafo conciso, formatado como um e-mail para o CEO.
```

3. **Criação de Tabela:**
```
[PERSONA: Você é um analista de dados sênior.]
[DADOS: [Lista de 5 produtos, seus preços e margens de lucro]]
[FORMATO: Use Markdown para a tabela.]
### AÇÃO FINAL ###
Liste os 5 produtos em uma tabela, ordenados pela margem de lucro decrescente.
```

4. **Brainstorming Estruturado:**
```
[TAREFA: Desenvolver 10 ideias de títulos para um artigo de blog sobre "Engenharia de Prompt para Iniciantes".]
[REQUISITOS: Os títulos devem ser cativantes e incluir um número.]
### AÇÃO FINAL ###
Gere a lista de 10 títulos e, em seguida, escolha o melhor e justifique sua escolha em uma frase.
```

5. **Instrução Condicional:**
```
[CONTEXTO: Analise o seguinte feedback do cliente: "O produto é bom, mas o preço é muito alto."]
[PERSONA: Você é um gerente de atendimento ao cliente.]
[FORMATO: Resposta em tom empático e profissional.]
### AÇÃO FINAL ###
Se o feedback for negativo, gere uma resposta pedindo desculpas e oferecendo um cupom de 10%. Caso contrário, apenas agradeça.
```

6. **Revisão e Edição:**
```
[TEXTO A REVISAR: [Parágrafo com erros gramaticais e de pontuação]]
[REQUISITOS: Mantenha o tom formal e corrija apenas a gramática e a pontuação.]
### AÇÃO FINAL ###
Reescreva o parágrafo corrigido.
```

7. **Criação de Roteiro:**
```
[TEMA: Vídeo curto para TikTok sobre os benefícios do café.]
[ESTRUTURA: 30 segundos, 3 cenas, com gancho nos primeiros 3 segundos.]
### AÇÃO FINAL ###
Escreva o roteiro completo, incluindo a descrição da cena e o diálogo.
```
```

## Best Practices
**Clareza e Especificidade:** A CTA deve ser a instrução mais clara e específica do prompt. Use verbos de ação fortes (e.g., "Gere", "Resuma", "Liste", "Execute"). **Posicionamento:** Coloque a CTA no final do prompt, após todo o contexto, persona e restrições. Isso garante que o LLM a processe como a instrução final de execução. **Formato de Saída:** Combine a CTA com uma instrução de formato de saída (e.g., "Gere a tabela em formato Markdown", "Responda em JSON"). **Isolamento:** Use marcadores visuais (como `### AÇÃO FINAL ###` ou tags XML) para isolar a CTA do restante do prompt, garantindo que o LLM a identifique como o comando de execução. **Teste e Iteração:** Se a saída não for a esperada, refine a CTA antes de alterar o contexto ou a persona.

## Use Cases
**Automatização de Tarefas:** Ideal para prompts que visam a execução de uma tarefa específica, como gerar código, criar um resumo ou traduzir um texto. **Estruturação de Saída:** Essencial para garantir que o LLM produza o resultado em um formato específico (e.g., JSON, XML, Markdown, tabela), facilitando o processamento posterior por outras ferramentas ou scripts. **Encadeamento de Prompts (Chain Prompting):** Atua como o comando final em uma etapa de um processo de encadeamento, garantindo que a saída seja o *input* exato necessário para a próxima etapa. **Minimização de Divagação:** Usado em prompts longos para evitar que o LLM comece a resposta com introduções ou explicações desnecessárias, indo direto ao ponto da execução. **Criação de Conteúdo Direcionado:** Garante que a peça de conteúdo final (e.g., e-mail, título, postagem em rede social) contenha o elemento de ação desejado.

## Pitfalls
**CTA Vaga ou Ambígua:** Usar CTAs como "Continue" ou "O que mais?" não fornece direção suficiente, levando a respostas genéricas. **Posicionamento Incorreto:** Colocar a CTA no início ou no meio do prompt pode fazer com que o LLM a execute prematuramente, ignorando o contexto subsequente. **Sobrecarga de Ações:** Incluir múltiplas ações não relacionadas na CTA (e.g., "Gere o código, escreva um poema e me diga o que você comeu no café da manhã") confunde o modelo. **Ausência de Formato:** Não especificar o formato de saída (e.g., lista, tabela, JSON) junto com a CTA pode resultar em uma resposta que executa a ação, mas em um formato difícil de usar. **Confundir CTA de Marketing com CTA de Prompt:** Usar linguagem de marketing (e.g., "Clique aqui para saber mais") em vez de comandos de execução (e.g., "Gere o resultado") dentro do prompt.

## URL
[https://www.reddit.com/r/PromptEngineering/comments/1ius9pt/my_favorite_prompting_technique_whats_yours/?tl=pt-br](https://www.reddit.com/r/PromptEngineering/comments/1ius9pt/my_favorite_prompting_technique_whats_yours/?tl=pt-br)
