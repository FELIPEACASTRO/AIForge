# Meta-Prompting

## Description

Meta-Prompting é uma técnica avançada de engenharia de prompt que utiliza um Large Language Model (LLM) de 'inteligência superior' para gerar, refinar e otimizar prompts para um LLM de 'inteligência inferior' ou para o próprio modelo. Diferentemente das técnicas tradicionais centradas no conteúdo, o Meta-Prompting prioriza considerações estruturais e sintáticas, transformando o processo de criação de prompts em uma tarefa de metaprogramação. Isso permite que o modelo decomponha tarefas complexas de raciocínio em subproblemas mais simples e demonstre fortes capacidades de seguir instruções através de aprendizado in-contexto (zero-shot), sem a necessidade de fine-tuning extensivo.

## Statistics

O Meta-Prompting demonstrou ganhos significativos em eficiência de tokens e desempenho em tarefas complexas. Em testes, o modelo Qwen-72B com Meta-Prompting (zero-shot) alcançou 46.3% de precisão PASS@1 em problemas MATH, superando o GPT-4 (2023-0314) que registrou 42.5%. No benchmark GSM8K, o mesmo modelo atingiu 83.5% de precisão, superando as melhores abordagens de few-shot prompting e modelos fine-tuned. Além disso, alcançou 100% de taxa de sucesso nas tarefas Game of 24 usando GPT-4.

## Features

As principais características incluem: 1. **Decomposição de Tarefas:** Capacidade de quebrar tarefas de raciocínio complexas em subproblemas mais gerenciáveis. 2. **Auto-Refinamento de Prompt:** Permite que o LLM gere e refine seus próprios prompts de forma recursiva. 3. **Eficiência de Tokens:** Melhora a eficiência do uso de tokens. 4. **Instrução Zero-Shot:** Elicita fortes capacidades de seguir instruções em modelos base grandes apenas com aprendizado in-contexto.

## Use Cases

Otimização automática de prompts, melhoria da qualidade e estrutura da saída do LLM, adaptação de prompts para diferentes modelos (onde um modelo mais forte otimiza para um modelo mais fraco), e resolução de problemas complexos de raciocínio e matemática.

## Integration

O Meta-Prompting é implementado usando um prompt de nível superior (o meta-prompt) que instrui o LLM a gerar ou modificar um prompt de nível inferior (o prompt de tarefa). \n\n**Exemplo de Meta-Prompt para Otimização de Resumo:**\n```\n\"\"\"\nMelhore o seguinte prompt para gerar um resumo mais detalhado.\nAdira às melhores práticas de engenharia de prompt.\nCertifique-se de que a estrutura seja clara e intuitiva e contenha o tipo de notícia, tags e sentimento.\n\n{prompt_simples}\n\nRetorne apenas o prompt melhorado.\n\"\"\"\n```\n\n**Prompt de Tarefa Refinado (Resultado do Meta-Prompt):**\n```\n'Por favor, leia o seguinte artigo de notícias e forneça um resumo abrangente que inclua:\\n\\n1. **Tipo de Notícia**: Especifique a categoria do artigo (ex: Política, Tecnologia, Saúde, Esportes, etc.).\\n2. **Resumo**: Escreva um resumo conciso e claro dos pontos principais, garantindo que a estrutura seja lógica e intuitiva.\\n3. **Tags**: Liste palavras-chave ou tags relevantes associadas ao artigo.\\n4. **Análise de Sentimento**: Analise o sentimento geral do artigo (positivo, negativo ou neutro) e explique brevemente seu raciocínio.\\n\\n**Artigo:**\\n\\n{artigo}'\n```

## URL

https://arxiv.org/html/2311.11482v7