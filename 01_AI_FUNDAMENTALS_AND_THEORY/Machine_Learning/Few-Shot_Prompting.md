# Few-Shot Prompting

## Description
O Few-Shot Prompting, ou Prompting de Poucos Disparos, é uma técnica fundamental de Engenharia de Prompt que explora o conceito de **Aprendizado no Contexto** (*In-Context Learning*) em Modelos de Linguagem Grande (LLMs). Diferentemente do Zero-Shot Prompting, onde apenas a instrução é fornecida, o Few-Shot Prompting envolve a inclusão de um pequeno número de exemplos de pares entrada-saída (os "shots") diretamente no prompt. Essas demonstrações servem para condicionar o modelo, mostrando o formato, o estilo e o tipo de resposta esperada para a tarefa. Esta técnica é particularmente eficaz para tarefas mais complexas ou específicas, onde o conhecimento pré-treinado do modelo não é suficiente, ou quando é necessário que a saída siga um formato ou estilo muito particular. A eficácia do Few-Shot Prompting foi observada pela primeira vez quando os modelos foram escalados para um tamanho suficiente, demonstrando a capacidade dos LLMs de aprender com exemplos fornecidos no contexto imediato da consulta.

## Examples
```
**Exemplo 1: Classificação de Sentimento (3-Shot)**

**Tarefa:** Classificar a opinião do cliente como "Positivo", "Negativo" ou "Neutro".

```
Comentário: O produto chegou com dois dias de atraso, mas a qualidade é excelente.
Sentimento: Neutro

Comentário: A interface é confusa e o suporte técnico não responde.
Sentimento: Negativo

Comentário: A melhor compra que fiz este ano! Recomendo a todos.
Sentimento: Positivo

Comentário: O preço é justo e a entrega foi no prazo, mas o manual é muito vago.
Sentimento:
```

**Saída Esperada:** Neutro

---

**Exemplo 2: Extração de Entidade (3-Shot)**

**Tarefa:** Extrair o nome do cliente e o número do pedido de um texto de suporte, formatando a saída como JSON.

```
Texto: Olá, sou a Ana Silva e meu pedido 45678 não chegou.
JSON: {"cliente": "Ana Silva", "pedido": "45678"}

Texto: Gostaria de saber sobre o status do pedido 12345. Meu nome é João Pereira.
JSON: {"cliente": "João Pereira", "pedido": "12345"}

Texto: O item que comprei, pedido 98765, veio errado. Falo em nome de Maria Souza.
JSON: {"cliente": "Maria Souza", "pedido": "98765"}

Texto: Por favor, verifiquem o pedido 54321. Meu nome é Carlos Eduardo.
JSON:
```

**Saída Esperada:** {"cliente": "Carlos Eduardo", "pedido": "54321"}

---

**Exemplo 3: Tradução com Estilo (2-Shot)**

**Tarefa:** Traduzir frases do Português para o Inglês, mantendo um tom **formal e corporativo**.

```
Português: A reunião foi adiada para a próxima semana.
Inglês: The meeting has been postponed until next week.

Português: Por favor, envie o relatório de progresso até o final do dia.
Inglês: Kindly submit the progress report by the close of business today.

Português: Precisamos de uma solução para otimizar nossos processos internos.
Inglês:
```

**Saída Esperada:** We require a solution to optimize our internal processes.

---

**Exemplo 4: Geração de Código (2-Shot)**

**Tarefa:** Gerar uma função Python que calcule a área de formas geométricas, seguindo o padrão de documentação e nomenclatura.

```
# Exemplo 1: Área do Quadrado
def calcular_area_quadrado(lado):
    """Calcula a área de um quadrado."""
    return lado * lado

# Exemplo 2: Área do Círculo
def calcular_area_circulo(raio):
    """Calcula a área de um círculo."""
    import math
    return math.pi * raio**2

# Exemplo 3: Área do Triângulo
def calcular_area_triangulo(base, altura):
    """Calcula a área de um triângulo."""
```

**Saída Esperada:**
```python
    return (base * altura) / 2
```

---

**Exemplo 5: Sumarização com Formato Específico (3-Shot)**

**Tarefa:** Resumir um parágrafo em uma única frase que comece com "Em suma,".

```
Parágrafo: A inteligência artificial generativa está revolucionando a criação de conteúdo, permitindo que máquinas produzam textos, imagens e músicas com qualidade cada vez maior. Isso tem implicações profundas para indústrias criativas e para a automação de tarefas rotineiras.
Resumo: Em suma, a IA generativa está transformando a criação de conteúdo e automatizando tarefas em diversas indústrias.

Parágrafo: O aquecimento global é um desafio complexo que exige a cooperação internacional, a transição para energias renováveis e a adoção de políticas de sustentabilidade rigorosas em todos os setores da economia.
Resumo: Em suma, o combate ao aquecimento global requer colaboração global e uma mudança urgente para práticas sustentáveis e energias limpas.

Parágrafo: A técnica de Few-Shot Prompting permite que modelos de linguagem aprendam um novo padrão de tarefa a partir de poucos exemplos fornecidos no prompt, melhorando significativamente o desempenho em tarefas específicas sem a necessidade de um novo treinamento.
Resumo:
```

**Saída Esperada:** Em suma, o Few-Shot Prompting aprimora o desempenho de LLMs em tarefas específicas ao fornecer poucos exemplos de aprendizado no contexto.
```

## Best Practices
1. **Consistência e Clareza:** Mantenha a formatação, a estrutura e o estilo dos exemplos **uniformes** e **consistentes**. O modelo aprende o padrão a partir dos exemplos, e qualquer inconsistência pode levar a saídas imprevisíveis. 2. **Gerenciamento de Tokens:** Priorize exemplos **concisos** e **diretos** para evitar exceder o limite da janela de contexto (*token limit*). Para padrões repetitivos, é mais eficiente resumir a regra do que incluir muitos exemplos longos. 3. **Número Ideal de Exemplos:** O número de "shots" deve ser determinado empiricamente, geralmente variando entre **2 a 5 exemplos**. Tarefas mais complexas podem exigir mais, mas é crucial encontrar um equilíbrio para não desperdiçar tokens ou confundir o modelo. 4. **Alinhamento com a Tarefa:** Os exemplos fornecidos devem ser **altamente relevantes** e se alinhar de perto com o tipo de entrada e saída esperados para a consulta final. Incluir exemplos que representem cenários desafiadores ou "casos de borda" pode melhorar a robustez do modelo. 5. **Foco na Tarefa:** Mantenha os prompts **específicos para a tarefa**. Evitar misturar tipos de tarefas (como classificação e sumarização) no mesmo prompt, a menos que as tarefas sejam claramente separadas e definidas.

## Use Cases
1. **Classificação de Texto:** Categorização de sentimentos (positivo/negativo), detecção de spam, ou classificação de tickets de suporte em categorias específicas. 2. **Sumarização e Extração de Informação:** Resumir textos longos em um formato específico (ex: uma única frase, ou um formato JSON), ou extrair entidades específicas de um texto. 3. **Tradução e Transcriação:** Tradução de frases em um estilo ou tom particular, ou tradução de termos técnicos que exigem um vocabulário específico. 4. **Geração de Código:** Fornecer exemplos de código para que o modelo gere novas funções ou scripts seguindo o mesmo padrão de sintaxe e estilo. 5. **Modelagem de Estilo e Tom:** Gerar conteúdo (marketing, escrita criativa) que imite um tom de voz específico (ex: formal, humorístico, técnico) ou um formato de documento (ex: e-mail, post de blog, tweet).

## Pitfalls
1. **Inconsistência de Formatação:** O erro mais comum é a variação na estrutura dos exemplos, o que impede o modelo de identificar o padrão de entrada-saída. 2. **Excesso ou Escassez de Exemplos:** Usar muitos exemplos pode levar ao estouro da janela de contexto e perda de informações, enquanto usar poucos pode resultar em desempenho subótimo. 3. **Mistura de Tarefas:** Tentar resolver tarefas fundamentalmente diferentes no mesmo prompt sem separação clara. 4. **Ignorar Limites de Token:** Não considerar o custo e o limite de tokens, especialmente em modelos com janelas de contexto menores. 5. **Confundir com Raciocínio Complexo:** O Few-Shot Prompting não é a solução ideal para tarefas que exigem **múltiplas etapas de raciocínio** (como problemas de matemática complexa ou lógica). Nesses casos, técnicas como o Chain-of-Thought Prompting (CoT) são mais apropriadas.

## URL
[https://www.promptingguide.ai/pt/techniques/fewshot](https://www.promptingguide.ai/pt/techniques/fewshot)
