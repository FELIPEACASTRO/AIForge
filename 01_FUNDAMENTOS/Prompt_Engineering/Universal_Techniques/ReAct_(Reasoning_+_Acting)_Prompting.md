# ReAct (Reasoning + Acting) Prompting

## Description

ReAct (Reasoning + Acting) é um *framework* de *prompting* que capacita Large Language Models (LLMs) a resolver tarefas complexas ao intercalar rastros de **Raciocínio** (*Thought*) e **Ações** (*Action*) específicas da tarefa. Essa sinergia permite que o raciocínio ajude o modelo a induzir, rastrear e atualizar planos de ação, enquanto as ações permitem que ele interaja com fontes externas (como APIs ou bases de conhecimento) para coletar informações adicionais. O ReAct supera problemas de alucinação e propagação de erros comuns em métodos puramente baseados em *Chain-of-Thought* (CoT), tornando as trajetórias de resolução de tarefas mais interpretáveis e confiáveis.

## Statistics

*   **Citações:** Mais de 5.355 citações (Artigo original de 2022, ICLR 2023). *   **Métricas de Desempenho (Sucesso Absoluto):** *   **ALFWorld (Tomada de Decisão Interativa):** Supera métodos de *Imitation Learning* e *Reinforcement Learning* em 34% na taxa de sucesso. *   **WebShop (Compra Online Interativa):** Supera *baselines* em 10% na taxa de sucesso. *   **Eficiência:** Alcança resultados de ponta com apenas um ou dois exemplos *in-context* (few-shot prompting).

## Features

1. **Intercalação Raciocínio-Ação:** O modelo alterna entre gerar um *Thought* (Raciocínio) e uma *Action* (Ação) seguida por uma *Observation* (Observação) do resultado da ação. 2. **Uso de Ferramentas (Tool Use):** Permite que o LLM utilize ferramentas externas (APIs, busca na web, código) para estender suas capacidades e acessar informações em tempo real. 3. **Planejamento Dinâmico:** O *Thought* ajuda a induzir, rastrear e atualizar o plano de ação, permitindo que o agente se adapte a situações inesperadas. 4. **Redução de Alucinação:** Ao buscar informações externas, o modelo se baseia em fatos verificáveis, reduzindo a tendência à alucinação. 5. **Interpretabilidade:** A sequência explícita de *Thought*, *Action* e *Observation* torna o processo de resolução de problemas mais transparente e rastreável.

## Use Cases

*   **Sistemas de Agentes (AI Agents):** É a base para a construção de agentes de IA capazes de realizar tarefas complexas e multi-etapas. *   **Resolução de Problemas Complexos:** Tarefas que exigem raciocínio lógico e acesso a informações externas, como responder a perguntas factuais que exigem pesquisa (HotpotQA, Fever). *   **Tomada de Decisão Interativa:** Ambientes como jogos baseados em texto (ALFWorld) ou plataformas de compra online (WebShop). *   **Função de Chamada (Function Calling):** Integração eficiente de ferramentas e APIs em fluxos de trabalho de LLMs. *   **Chatbots e Assistentes Virtuais:** Criação de assistentes que podem raciocinar sobre a intenção do usuário e executar ações como calcular, pesquisar ou interagir com sistemas internos (ex: *chatbots* bancários).

## Integration

O ReAct é implementado através de um *prompt* que define o formato de saída esperado, geralmente um ciclo **Thought-Action-Observation (TAO)**. **Estrutura do Prompt (Exemplo Genérico):** ``` Você é um agente de IA que pode usar as seguintes ferramentas: [LISTA DE FERRAMENTAS]. Para responder à pergunta, você deve seguir o formato Thought/Action/Observation. Use Thought para raciocinar sobre o próximo passo. Use Action para chamar uma ferramenta (ex: Action: search[termo de busca]). Use Observation para registrar o resultado da ferramenta. O ciclo termina quando você tiver a resposta final, que deve ser fornecida após o último Thought. Pergunta: [PERGUNTA DO USUÁRIO] Thought: [Raciocínio inicial sobre a pergunta e a ferramenta a ser usada] Action: [Chamada da ferramenta e argumento] Observation: [Resultado da ferramenta] Thought: [Raciocínio sobre a Observation e o próximo passo] Action: [Chamada da ferramenta e argumento, se necessário] Observation: [Resultado da ferramenta, se necessário] Thought: [Raciocínio final e formulação da resposta] Answer: [RESPOSTA FINAL] ``` **Melhores Práticas:** *   **Estrutura Clara:** Sempre rotule explicitamente *Thought*, *Action* e *Observation* para guiar o modelo. *   **Definição de Ferramentas:** Forneça uma lista clara e concisa das ferramentas disponíveis e seus formatos de uso. *   **Instruções de Parada:** Defina claramente como o modelo deve terminar o ciclo (ex: com a *tag* `Answer:`). *   **Few-Shot Examples:** Incluir um ou dois exemplos completos de interações TAO no *prompt* melhora drasticamente o desempenho.

## URL

https://arxiv.org/abs/2210.03629