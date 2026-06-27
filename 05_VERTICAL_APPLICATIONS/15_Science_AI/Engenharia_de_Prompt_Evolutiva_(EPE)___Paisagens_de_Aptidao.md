# Engenharia de Prompt Evolutiva (EPE) / Paisagens de Aptidão

## Description
A Engenharia de Prompt Evolutiva (EPE) é uma metodologia de otimização de prompts que aplica os princípios da biologia evolutiva, notavelmente a metáfora da **Paisagem de Aptidão (Fitness Landscape)** de Sewall Wright, ao espaço de prompts. Nesta abordagem, um prompt é tratado como um 'genótipo' ou 'fenótipo', e sua 'aptidão' (fitness) é determinada pela performance do Large Language Model (LLM) na tarefa desejada (ex: acurácia, coerência). Algoritmos evolutivos, como Algoritmos Genéticos (AGs), são usados para gerar, mutar e selecionar prompts de forma iterativa, permitindo que o processo de otimização navegue em espaços de prompts complexos e 'acidentados' (rugged landscapes), onde pequenas mudanças no prompt podem levar a grandes variações de desempenho. O objetivo é encontrar o prompt de 'aptidão' máxima, ou seja, o prompt que extrai o melhor desempenho do LLM para uma tarefa específica.

## Examples
```
1. **Prompt Inicial (População Zero):** 'Gere uma lista de 10 sinônimos para a palavra [PALAVRA].'
2. **Prompt Mutado (Melhoria de Clareza):** 'Você é um lexicógrafo. Sua tarefa é gerar uma lista concisa e precisa de 10 sinônimos de alta frequência para o termo [PALAVRA]. Use apenas palavras de uso comum.'
3. **Prompt Mutado (Adição de Restrição):** 'Gere 10 sinônimos para [PALAVRA]. **Restrição:** Todos os sinônimos devem ter 5 letras ou mais. **Formato:** Lista numerada.'
4. **Prompt Otimizado (DEEVO - Debate-Driven Evolutionary Prompt Optimization):** 'Instrução: Atue como um especialista em detecção de erros. Analise o seguinte código [CÓDIGO] e identifique o erro de sintaxe e a linha exata. Em seguida, forneça a correção. **Critério de Aptidão:** A correção deve passar em 95% dos testes unitários. **Formato de Saída:** JSON com chaves 'erro', 'linha', 'correção'.'
5. **Prompt Otimizado (GAAPO - Genetic Algorithm Applied to Prompt Optimization):** 'Tarefa: Classifique o sentimento do texto [TEXTO] como Positivo, Negativo ou Neutro. **Contexto:** Considere a ironia. **Exemplo:** [EXEMPLO DE TEXTO E CLASSIFICAÇÃO]. **Meta:** Maximize a pontuação F1. **Instrução Adicional:** Pense passo a passo antes de responder.'
```

## Best Practices
1. **Definir uma Função de Aptidão Clara:** A métrica de desempenho (aptidão) deve ser objetiva e mensurável (ex: acurácia, F1-score, taxa de conclusão de tarefa) para guiar a evolução de forma eficaz.
2. **Manter a Diversidade da População:** Use operadores de mutação e cruzamento que explorem o espaço de prompts de forma ampla (diversificação orientada à novidade) para evitar ótimos locais.
3. **Usar Modelos de Linguagem como Operadores:** Em vez de mutações aleatórias de strings, use um LLM para gerar 'mutações' semânticas (ex: 'Reescreva este prompt para ser mais conciso', 'Adicione uma restrição de formato').
4. **Incorporar Mecanismos de Debate/Reflexão:** Técnicas como DEEVO (Debate-Driven Evolutionary Prompt Optimization) usam múltiplos agentes LLM para avaliar e refinar prompts, aumentando a robustez da seleção.
5. **Aproveitar a Estrutura da Paisagem:** Se a tarefa for complexa (paisagem acidentada), prefira algoritmos evolutivos. Se for simples (paisagem suave), a otimização incremental manual pode ser suficiente.

## Use Cases
1. **Otimização Automatizada de Prompts:** Encontrar o prompt de melhor desempenho para tarefas específicas (ex: classificação de texto, extração de entidades) sem intervenção humana contínua.
2. **Geração de Prompts Robustos:** Criar prompts que mantenham um alto desempenho mesmo com pequenas variações na entrada ou no modelo subjacente.
3. **Pesquisa em LLMs:** Estudar a sensibilidade e a topologia do espaço de prompts de diferentes modelos, revelando como eles respondem a variações semânticas.
4. **Aplicações de Código:** Otimizar prompts para tarefas de geração e correção de código (ex: EPiC - Evolutionary Prompt Engineering for Code).
5. **Sistemas RAG (Retrieval-Augmented Generation):** Otimizar o prompt de consulta para melhorar a recuperação de documentos e a qualidade da resposta final.

## Pitfalls
1. **Função de Aptidão Mal Definida:** Uma métrica de aptidão subjetiva ou imprecisa pode levar à evolução de prompts que parecem bons, mas não resolvem o problema real.
2. **Convergência Prematura:** A população de prompts pode se prender a um ótimo local (local optima) e parar de explorar o espaço, falhando em encontrar o melhor prompt global.
3. **Custo Computacional Elevado:** A avaliação de cada prompt (cálculo da aptidão) requer uma chamada ao LLM, tornando o processo evolutivo caro e lento se a população for grande ou o número de gerações for alto.
4. **Mutação Não Semântica:** Mutações aleatórias de strings podem gerar prompts sintaticamente inválidos ou sem sentido, desperdiçando recursos de avaliação.
5. **Ignorar a Topologia da Paisagem:** Usar otimização incremental (hill-climbing) em uma paisagem acidentada, o que quase garante que o processo ficará preso em um ótimo local subótimo.

## URL
[https://arxiv.org/html/2509.05375v1](https://arxiv.org/html/2509.05375v1)
