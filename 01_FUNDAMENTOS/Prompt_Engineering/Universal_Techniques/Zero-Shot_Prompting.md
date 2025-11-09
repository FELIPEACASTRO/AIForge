# Zero-Shot Prompting

## Description

Uma técnica fundamental de Prompt Engineering onde o Large Language Model (LLM) é instruído a completar uma tarefa sem receber exemplos prévios (demonstrações) no prompt. O modelo depende inteiramente de seu conhecimento interno adquirido durante o pré-treinamento para inferir a resposta apropriada. É a forma mais simples de prompting e serve como linha de base para técnicas mais avançadas como Few-Shot e Chain-of-Thought. A eficácia do Zero-Shot Prompting é um testemunho do poder dos modelos de linguagem modernos, que são ajustados para seguir instruções e generalizar conhecimento para tarefas não vistas.

## Statistics

A precisão em tarefas de raciocínio pode ser significativamente aumentada com variações. Por exemplo, o **Role-Play Prompting** (uma variação zero-shot) elevou a precisão do ChatGPT em AQuA de 53.5% para 63.8% e em Last Letter de 23.8% para 84.2% [3]. O Zero-Shot Prompting é a base para o **Zero-Shot-CoT** (Chain-of-Thought), que instrui o modelo a "pensar passo a passo" para melhorar o desempenho em tarefas complexas de raciocínio. A técnica é amplamente citada em pesquisas de 2023 e 2024 como um método fundamental para avaliar a capacidade de generalização de LLMs [1] [2].

## Features

- **Simplicidade:** Não requer a criação de exemplos de treinamento (demonstrações) no prompt.
- **Generalização:** Aproveita a capacidade do LLM de generalizar conhecimento de seu pré-treinamento para novas tarefas.
- **Linha de Base:** Serve como o método de prompting mais básico e uma linha de base para avaliar o desempenho de técnicas mais complexas.
- **Variantes Avançadas:** Pode ser combinado com outras técnicas zero-shot, como **Role-Play Prompting** e **Zero-Shot-CoT** (Chain-of-Thought), para melhorar o raciocínio e a precisão.

## Use Cases

- **Classificação de Texto:** Determinar o sentimento, tópico ou intenção de um texto.
- **Tradução:** Traduzir um texto de um idioma para outro.
- **Resumo:** Gerar um resumo conciso de um documento ou artigo.
- **Geração de Código:** Criar pequenos trechos de código com base em uma descrição.
- **Resposta a Perguntas (QA):** Responder a perguntas factuais ou conceituais diretamente.
- **Raciocínio Simples:** Resolver problemas lógicos ou matemáticos básicos.

## Integration

**Exemplo Básico (Classificação de Sentimento):**
`Classifique o texto em neutro, negativo ou positivo.
Texto: Eu acho que as férias foram razoáveis.
Sentimento:`

**Exemplo Avançado (Role-Play Prompting):**
`Você é um especialista em lógica e raciocínio. Analise a seguinte questão e forneça a resposta mais precisa.
Questão: Se todos os A são B, e todos os B são C, qual é a relação entre A e C?`

**Melhores Práticas:**
1. **Clareza e Especificidade:** A instrução deve ser o mais clara e específica possível, definindo a tarefa e o formato de saída desejado.
2. **Uso de Palavras-Chave:** Incluir palavras-chave que o modelo associa à tarefa (ex: "resumir", "traduzir", "classificar").
3. **Definição de Papel (Role-Play):** Atribuir um papel ao modelo (ex: "Você é um especialista...", "Aja como um tradutor profissional...") pode melhorar o desempenho em tarefas de raciocínio e geração.

## URL

https://www.promptingguide.ai/techniques/zeroshot