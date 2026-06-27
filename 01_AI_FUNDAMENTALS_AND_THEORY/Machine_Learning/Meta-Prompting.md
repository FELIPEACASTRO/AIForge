# Meta-Prompting

## Description
**Meta-Prompting** é uma técnica avançada de Engenharia de Prompt que se concentra nos aspectos estruturais e sintáticos das tarefas, em vez de detalhes específicos do conteúdo. Em sua definição formal, um Meta Prompt é um prompt estruturado, agnóstico a exemplos, que fornece um andaime para capturar a estrutura de raciocínio de uma categoria de tarefas, focando no *como* o problema deve ser resolvido, e não no *o quê* [1]. Em um contexto prático, a técnica envolve o uso de um Modelo de Linguagem Grande (LLM), geralmente um mais capaz (o Meta-LLM), para gerar, refinar ou otimizar prompts para outro LLM (o LLM alvo), garantindo que o prompt final seja mais eficaz e force uma saída de alta qualidade e estrutura consistente [2]. As principais características incluem ser orientado à estrutura, focado na sintaxe, usar exemplos abstratos e ter uma abordagem categórica, o que o torna mais eficiente em termos de tokens e mais justo para comparação de modelos do que o Few-Shot Prompting [3].

## Examples
```
**Exemplo 1: Otimização de Prompt (Geral)**

**Meta-Prompt (Input para o Meta-LLM):**
```
"""
Você é um Engenheiro de Prompt de nível sênior. Sua tarefa é analisar o 'PROMPT INICIAL' abaixo e reescrevê-lo para maximizar a qualidade e a criatividade da resposta de um LLM.

Instruções para a reescrita:
1. Adicione uma persona detalhada (ex: 'escritor de ficção científica premiado').
2. Defina o formato de saída (ex: 'um conto de 500 palavras').
3. Inclua restrições de estilo (ex: 'tom melancólico, com reviravolta no final').
4. O prompt final deve ser autocontido e não deve conter estas instruções.

PROMPT INICIAL:
"Escreva sobre um robô que se sente sozinho."

Retorne apenas o prompt otimizado.
"""
```

**Exemplo 2: Estrutura de Saída (JSON Schema)**

**Meta-Prompt (Input para o Meta-LLM):**
```
"""
Crie um prompt para um LLM que garanta que a saída seja uma lista de tarefas (to-do list) para um projeto de desenvolvimento de software, estritamente no formato JSON.

O prompt deve incluir:
1. A instrução para usar o formato JSON.
2. O esquema JSON obrigatório: um array de objetos, onde cada objeto tem as chaves 'id' (integer), 'tarefa' (string), 'prioridade' (string: 'Alta', 'Média', 'Baixa') e 'status' (string: 'Pendente', 'Em Progresso', 'Concluído').
3. A tarefa a ser processada: "Planejar o lançamento de um novo aplicativo de produtividade."

Retorne apenas o prompt final, incluindo o esquema JSON como parte das instruções.
"""
```

**Exemplo 3: Decomposição de Tarefas (Chain-of-Thought Forçado)**

**Meta-Prompt (Input para o Meta-LLM):**
```
"""
Crie um prompt que instrua o LLM a resolver o seguinte problema de lógica, utilizando obrigatoriamente a técnica de Chain-of-Thought (CoT) antes de fornecer a resposta final.

Problema: "Se um trem viaja a 60 km/h e percorre 300 km, e um segundo trem viaja a 90 km/h e percorre 450 km, qual trem chegou primeiro se ambos partiram ao mesmo tempo?"

O prompt deve ter duas seções claras: 'Passos de Raciocínio' e 'Resposta Final'.
"""
```

**Exemplo 4: Criação de Persona e Restrições de Linguagem**

**Meta-Prompt (Input para o Meta-LLM):**
```
"""
Gere um prompt para um LLM que o instrua a atuar como um 'Consultor de Segurança Cibernética Sênior'.

O prompt deve impor as seguintes restrições:
1. O tom deve ser extremamente formal e técnico.
2. O vocabulário deve ser especializado em segurança da informação (ex: 'criptografia', 'vetor de ataque', 'zero-day').
3. A tarefa é: "Explicar os riscos de segurança de uma rede Wi-Fi pública."

Retorne apenas o prompt de persona e tarefa.
"""
```

**Exemplo 5: Meta-Prompting para Geração de Prompt Multilíngue**

**Meta-Prompt (Input para o Meta-LLM):**
```
"""
Traduza e otimize o 'PROMPT INICIAL' em inglês para o português.

Otimização obrigatória:
1. O prompt em português deve instruir o LLM a formatar a resposta final em uma tabela Markdown.
2. A tabela deve ter colunas para 'Conceito' e 'Definição'.

PROMPT INICIAL:
"Explain the concept of quantum entanglement in simple terms."

Retorne apenas o prompt otimizado em português.
"""
```
```

## Best Practices
**Foco na Estrutura e Sintaxe**: Priorize a definição de formato, persona e restrições de saída (JSON, tabelas, CoT) no Meta-Prompt, em vez de apenas o conteúdo. **Utilize um Meta-LLM Superior**: Sempre que possível, use um modelo de linguagem mais avançado e capaz (o Meta-LLM) para gerar ou refinar prompts para modelos menos potentes, garantindo maior qualidade e complexidade no prompt final. **Clareza e Especificidade**: O Meta-Prompt deve ser extremamente claro sobre o objetivo da otimização e as restrições que o prompt gerado deve impor ao LLM alvo. **Decomposição de Tarefas**: Use o Meta-Prompting para forçar a decomposição de problemas complexos em etapas de raciocínio claras (como o Chain-of-Thought), melhorando a precisão da resposta.

## Use Cases
**Otimização de Prompt**: Geração de prompts mais eficazes e detalhados a partir de prompts simples ou vagos. **Padronização de Saída**: Forçar o LLM alvo a produzir respostas em formatos estritos e consistentes, como JSON, XML ou tabelas Markdown, essencial para integração com sistemas de software. **Decomposição de Problemas Complexos**: Estruturar o prompt para obrigar o LLM a seguir um processo de raciocínio passo a passo (Chain-of-Thought), melhorando a precisão em tarefas de lógica, matemática e codificação. **Criação de Prompts para Múltiplos Modelos**: Gerar prompts otimizados para diferentes modelos de LLM (ex: um prompt para um modelo rápido e outro para um modelo mais lento, mas mais preciso). **Refinamento de Persona e Estilo**: Definir e impor persona, tom e vocabulário específicos para a resposta do LLM alvo.

## Pitfalls
**Meta-Prompt Vago**: Se o Meta-Prompt não for específico sobre o objetivo da otimização ou as restrições de saída, o prompt gerado pode não ser significativamente melhor do que o original. **Custo e Latência Elevados**: O uso de um LLM mais capaz (Meta-LLM) para gerar o prompt aumenta o custo e a latência da chamada total, pois são necessárias duas chamadas de API (uma para gerar o prompt e outra para executá-lo). **Dependência do Meta-LLM**: A qualidade do prompt final é altamente dependente da capacidade e do desempenho do LLM usado para gerar o Meta-Prompt. **Conhecimento Inato Presumido**: A técnica assume que o LLM alvo possui o conhecimento inato necessário para a tarefa; o desempenho pode deteriorar em tarefas muito únicas ou novas, assemelhando-se ao Zero-Shot Prompting [3].

## URL
[https://arxiv.org/html/2311.11482v7](https://arxiv.org/html/2311.11482v7)
