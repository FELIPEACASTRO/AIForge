# Few-Shot Prompting

## Description

Few-Shot Prompting (FSP) é uma técnica de Engenharia de Prompt que envolve fornecer ao Large Language Model (LLM) um pequeno número de exemplos (tipicamente 2 a 5) diretamente no prompt de entrada. Esses exemplos demonstram o formato e o estilo de resposta desejados, permitindo que o modelo utilize o que é conhecido como **Aprendizagem em Contexto (In-Context Learning - ICL)**. O modelo reconhece o padrão, a tarefa e o formato a partir desses exemplos e aplica essa lógica à nova entrada, sem a necessidade de qualquer ajuste fino (fine-tuning) ou atualização de parâmetros. Esta técnica foi popularizada pelo artigo seminal do GPT-3 da OpenAI em 2020, que demonstrou que modelos em larga escala podem aprender tarefas a partir de poucas demonstrações textuais [1].

## Statistics

A FSP demonstrou melhorias significativas de desempenho em relação às abordagens Zero-Shot para tarefas complexas. Estudos de caso mostram: redução de 60% no tempo de triagem manual de tickets de suporte e melhoria de 92% na precisão do roteamento [1]. Para tarefas de geração de conteúdo, a FSP pode reduzir o tempo de criação e edição em até 40% e alcançar 85% de consistência no tom [1]. Pesquisas recentes (2024-2025) indicam que, para modelos de raciocínio avançado (como o1 ou R1), a FSP pode, na verdade, degradar o desempenho em comparação com o Zero-Shot, sugerindo que a eficácia é dependente do modelo [1].

## Features

Aprendizagem em Contexto (ICL); Não requer Fine-Tuning; Alta Flexibilidade (exemplos podem ser alterados rapidamente); Melhora a precisão e a consistência do formato de saída; Ideal para tarefas que exigem um tom ou estilo específico; Permite a demonstração de casos de borda (edge cases) e exceções [1].

## Use Cases

Classificação de Texto (Análise de Sentimento, Detecção de Spam, Categorização de Tópicos); Extração e Transformação de Dados (Análise de faturas, Parsing de currículos, Normalização de endereços); Geração de Código (Adesão a convenções de codificação e estilo de documentação); Criação de Conteúdo (Geração de texto que corresponde à voz da marca); Tradução de Linguagem (Manutenção de terminologia específica do domínio); Automação de Atendimento ao Cliente (Roteamento de tickets, Geração de modelos de resposta) [1].

## Integration

Melhores Práticas:\n1. **Qualidade sobre Quantidade:** Use exemplos diversos que cubram diferentes aspectos da tarefa, em vez de muitos exemplos semelhantes. Cada exemplo deve ser único [1].\n2. **Formato Consistente:** Mantenha um formato de entrada/saída uniforme em todos os exemplos (ex: `Input: 'Texto' -> Output: 'Rótulo'`) [1].\n3. **Número de Exemplos:** Comece com 2-3 exemplos. Use 4-5 para tarefas mais complexas ou casos de borda. Evite exceder 8-10 exemplos, pois isso pode levar a retornos decrescentes e consumo excessivo de tokens [1].\n4. **Posicionamento:** Teste colocar o exemplo mais crítico ou difícil por último, pois alguns LLMs dão mais peso aos exemplos recentes (viés de recenticidade) [1].\n5. **Modelos de Raciocínio:** Para modelos avançados (ex: o1, R1), comece com Zero-Shot e adicione no máximo 1-2 exemplos apenas se necessário, pois a FSP pode prejudicar seu desempenho de raciocínio inerente [1].\n\nExemplo de Prompt (Classificação de Sentimento):\n```\nClassifique o sentimento do texto como Positivo, Negativo ou Neutro.\n\nExemplo 1: O produto superou minhas expectativas! -> Positivo\nExemplo 2: A entrega atrasou e o item veio danificado. -> Negativo\nExemplo 3: Funciona como esperado, sem surpresas. -> Neutro\n\nClassifique o sentimento do texto a seguir: O atendimento ao cliente foi excelente, mas o software é lento. ->\n```

## URL

https://www.articsledge.com/post/few-shot-prompting