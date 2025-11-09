# Prompt Templates (Modelos de Prompt)

## Description

Os Prompt Templates representam uma evolução na Engenharia de Prompt, substituindo a criação de prompts únicos e ineficientes por um arsenal de prompts reutilizáveis e estruturados. Eles funcionam como um 'andaime cognitivo', fornecendo uma estrutura consistente para guiar o Large Language Model (LLM) em tarefas específicas, garantindo resultados mais previsíveis e de maior qualidade. Um template é essencialmente um prompt salvo que inclui variáveis, permitindo que seja executado com diferentes opções de entrada (variáveis) sem a necessidade de reescrever a estrutura do prompt.

## Statistics

Pesquisas recentes (e.g., arXiv 2411.10541v1) indicam que a formatação do prompt pode ter um **impacto significativo no desempenho do modelo**, contrariando a suposição de estabilidade. O uso de templates é fundamental para métricas de avaliação, como o 'prompt alignment', que mede a aderência da resposta do LLM às instruções do template. A otimização contínua dos templates, baseada em métricas de desempenho e feedback do usuário, é uma prática recomendada para ambientes de produção.

## Features

Reutilização e Consistência; Estrutura Variável (uso de placeholders); Aplicação em Escala; Redução da Ineficiência de Prompts 'One-off'; Suporte a Governança e Padronização; Componente chave em ferramentas de avaliação de LLMs (e.g., Google Vertex AI, AWS Bedrock).

## Use Cases

Inovação e Desenvolvimento de Produtos (Geração de ideias, identificação de 'killer features', proposição de valor); Rotinas Corporativas (Criação de relatórios, análise de dados, geração de apresentações); Avaliação de Modelos (Definição de métricas de avaliação customizadas); Marketing e Vendas (Mensagens de marketing, pitches de inovação); Análise de Dados (Templates para análise financeira e interpretação de dados).

## Integration

Melhores Práticas: 1. **Estrutura Clara:** Coloque as instruções no início e use delimitadores (e.g., ### ou \"\"\" ) para separar a instrução do contexto. 2. **Especificidade:** Seja claro e específico sobre a tarefa e o formato de saída esperado. 3. **Variáveis:** Utilize variáveis (placeholders) para os dados de entrada, tornando o template genérico e reutilizável. 4. **Iteração:** Analise continuamente o desempenho e refine o template com base em falhas e feedback. Exemplo de Template (Análise de Dados): 'Você é um analista de dados especialista. Sua tarefa é [TAREFA_ESPECÍFICA]. Analise os dados fornecidos em [DADOS_DE_ENTRADA] e forneça um resumo conciso, seguido por 3 insights acionáveis. Formato de saída: JSON.'

## URL

https://mitsloan.mit.edu/ideas-made-to-matter/prompt-engineering-so-2024-try-these-prompt-templates-instead