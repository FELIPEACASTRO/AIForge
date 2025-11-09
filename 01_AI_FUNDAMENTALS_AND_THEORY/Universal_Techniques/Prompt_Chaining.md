# Prompt Chaining

## Description
**Prompt Chaining** (Encadeamento de Prompts) é uma técnica avançada de Engenharia de Prompt que consiste em decompor uma tarefa complexa em uma sequência de subtarefas mais simples e gerenciáveis. O resultado (output) de um prompt (ou uma etapa) é usado como a entrada (input) para o prompt subsequente, criando um fluxo de trabalho sequencial. Essa abordagem é fundamental para melhorar a **qualidade**, a **confiabilidade** e a **rastreabilidade** (observabilidade) das respostas de Modelos de Linguagem Grande (LLMs) em tarefas multifacetadas. Ao isolar o foco cognitivo do modelo em cada etapa, o Prompt Chaining reduz a carga cognitiva, minimiza erros de raciocínio (alucinações) e permite a aplicação de mecanismos de refinamento iterativo, espelhando fluxos de trabalho humanos de rascunho, crítica e revisão [1] [2]. É a base para estratégias como o *Least-to-Most Prompting* e é crucial para sistemas de produção que exigem alta precisão e monitoramento detalhado.

## Examples
```
**Exemplo 1: Síntese de Pesquisa e Geração de Artigo**
1. **Prompt 1 (Extração):** "Analise o texto a seguir e extraia os 5 pontos-chave mais importantes, formatando-os como uma lista JSON com o título 'PontosChave'."
2. **Prompt 2 (Análise):** "Com base nos 'PontosChave' extraídos (INPUT), identifique a principal tese e o público-alvo ideal para um artigo sobre o tema. Responda em formato JSON com as chaves 'TesePrincipal' e 'PublicoAlvo'."
3. **Prompt 3 (Geração):** "Usando a 'TesePrincipal' e o 'PublicoAlvo' (INPUT), escreva a introdução de um artigo de 500 palavras em tom profissional e persuasivo."

**Exemplo 2: Classificação e Resumo de E-mail**
1. **Prompt 1 (Classificação):** "Classifique o e-mail a seguir (INPUT) em uma das categorias: 'Urgente', 'Informativo', 'Financeiro', 'Marketing'. Responda apenas com a categoria."
2. **Prompt 2 (Resumo Condicional):** "Se a categoria do e-mail for 'Urgente' (INPUT), gere um resumo de uma frase com a ação necessária. Caso contrário, gere um resumo de um parágrafo."

**Exemplo 3: Refatoração de Código**
1. **Prompt 1 (Análise de Código):** "Analise o trecho de código Python a seguir (INPUT) e liste 3 áreas de melhoria de performance ou legibilidade. Responda em formato de lista numerada."
2. **Prompt 2 (Refatoração):** "Com base nas áreas de melhoria listadas (INPUT), refatore o código original para incorporar essas mudanças. Mantenha a funcionalidade original."

**Exemplo 4: Criação de Personagem para História**
1. **Prompt 1 (Conceito):** "Gere um conceito de personagem para uma história de fantasia, incluindo 'Nome', 'Raça' e 'Ocupação'. Responda em JSON."
2. **Prompt 2 (Detalhes):** "Usando o 'Nome' e a 'Raça' (INPUT), escreva um parágrafo detalhando a história de fundo (background) e a principal motivação do personagem."

**Exemplo 5: Análise de Sentimento e Resposta**
1. **Prompt 1 (Sentimento):** "Analise o comentário de cliente a seguir (INPUT) e classifique o sentimento como 'Positivo', 'Neutro' ou 'Negativo'. Responda apenas com o sentimento."
2. **Prompt 2 (Resposta):** "Com base no sentimento ('Negativo') (INPUT), escreva uma resposta de atendimento ao cliente empática e proponha uma solução. Se o sentimento for 'Positivo', escreva um agradecimento breve."
```

## Best Practices
**1. Decomposição Modular:** Divida a tarefa em etapas lógicas e discretas. Cada prompt na cadeia deve ter um objetivo único e bem definido. **2. Contratos de Dados Explícitos:** Defina um formato de saída (schema) estrito para cada prompt (preferencialmente JSON ou XML) para garantir que o output seja um input limpo e previsível para o próximo passo. **3. Minimizar o Contexto:** Passe apenas a informação essencial para o próximo prompt. Contexto excessivo aumenta o custo (tokens) e pode introduzir ruído ou desvio de foco. **4. Validação Intermediária:** Implemente verificações (determinísticas ou baseadas em LLM-as-a-judge) após cada etapa para garantir a qualidade e a conformidade do output antes de prosseguir. **5. Iteração e Refinamento:** Use a cadeia para simular um ciclo de *drafting*, *critique* e *revision*, onde um prompt avalia o output do anterior e o prompt seguinte o refina.

## Use Cases
**1. Tarefas Multi-Instrução:** Decompor tarefas que combinam extração, transformação, análise e visualização de dados em etapas ordenadas. **2. Fluxos de Trabalho de Agentes (Agents):** É a espinha dorsal de arquiteturas de agentes autônomos, onde o modelo planeja, executa e reflete sobre as ações em um ciclo iterativo. **3. Síntese e Revisão de Documentos:** Criar um rascunho, criticar o rascunho em relação a um conjunto de regras e, em seguida, refinar o rascunho final. **4. Raciocínio Complexo (Chain-of-Thought Aprimorado):** Usar o output de um prompt para forçar o modelo a gerar um passo de raciocínio explícito (e.g., "Pense passo a passo") e usar esse raciocínio como input para a resposta final. **5. Geração de Conteúdo Estruturado:** Criar um esboço (outline), gerar o conteúdo de cada seção e, por fim, revisar e formatar o documento completo.

## Pitfalls
**1. Explosão de Custo (Token Sprawl):** O encadeamento aumenta o número total de tokens processados, pois o output de cada etapa é reintroduzido como input. Isso pode levar a um aumento significativo no custo e na latência. **2. Erro em Cascata (Error Propagation):** Um erro ou alucinação em um prompt inicial é passado para os prompts subsequentes, contaminando toda a cadeia e levando a um resultado final incorreto. **3. Latência Excessiva:** A execução sequencial de múltiplos prompts aumenta a latência total da resposta. Para tarefas em tempo real, isso pode ser inaceitável. **4. Contexto Insuficiente:** A tentativa de minimizar o contexto para economizar tokens pode resultar na perda de informações cruciais necessárias para o raciocínio nas etapas seguintes. **5. Dependência Rígida de Schema:** Se o formato de saída (schema) de um prompt falhar, a cadeia pode quebrar. É crucial ter mecanismos robustos de validação e *retry* (tentativa) para lidar com falhas de formatação.

## URL
[https://www.getmaxim.ai/articles/prompt-chaining-for-ai-engineers-a-practical-guide-to-improving-llm-output-quality/](https://www.getmaxim.ai/articles/prompt-chaining-for-ai-engineers-a-practical-guide-to-improving-llm-output-quality/)
