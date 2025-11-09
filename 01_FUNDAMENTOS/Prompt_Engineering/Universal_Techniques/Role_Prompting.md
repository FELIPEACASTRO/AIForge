# Role Prompting

## Description

O Role Prompting (ou 'Prompting de Papel') é uma técnica de Engenharia de Prompt que consiste em atribuir uma **persona** ou um **papel específico** (como 'professor', 'historiador' ou 'vendedor') a um Large Language Model (LLM). O objetivo é guiar o estilo, o tom, o foco e o conhecimento da resposta do modelo, forçando-o a operar dentro de um contexto de domínio ou perspectiva específica. Isso melhora a relevância e a qualidade da saída para tarefas especializadas.

## Statistics

Pesquisas indicam que a técnica pode **melhorar a precisão e o raciocínio** em tarefas específicas, mas os resultados variam e dependem da qualidade da representação do papel nos dados de treinamento do modelo. Sugere-se que o uso de papéis não íntimos, termos neutros em relação ao gênero e papéis específicos para o público-alvo tendem a produzir melhores resultados. A eficácia é limitada pela extensão da pesquisa e pelos modelos específicos utilizados, podendo haver reforço de vieses e estereótipos se o papel for mal representado.

## Features

As principais características incluem a **melhoria da clareza e precisão** do texto ao alinhar a resposta com o papel atribuído. É útil para uma ampla gama de tarefas, como escrita, raciocínio e aplicações baseadas em diálogo, permitindo a personalização das respostas para se adequarem a contextos específicos.

## Use Cases

Aplicações em que a perspectiva ou o conhecimento especializado é crucial, como: **Assistência à Escrita** (ex: 'Você é um poeta'), **Raciocínio e Explicação** (ex: 'Você é um físico'), **Aplicações Baseadas em Diálogo** (ex: 'Você é um chatbot de suporte técnico') e **Personalização de Conteúdo** (ex: 'Você é um guia turístico de Paris').

## Integration

A melhor prática envolve uma **abordagem de duas etapas**:\n1. **Atribuição do Papel:** Comece com uma instrução clara e direta. Exemplo: 'Você é um historiador especializado na Revolução Francesa.'\n2. **Definição da Tarefa:** Em seguida, especifique a pergunta ou tarefa dentro desse contexto. Exemplo: 'Descreva as principais causas da queda da Bastilha.'\n\n**Exemplo de Prompt:**\n`Você é um desenvolvedor Python experiente. Sua tarefa é escrever uma função que calcule o fatorial de um número inteiro positivo de forma recursiva. Inclua docstrings e type hints.`

## URL

https://learnprompting.org/docs/advanced/zero_shot/role_prompting