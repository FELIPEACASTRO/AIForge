# Physics Problem Solving Prompts

## Description
A Engenharia de Prompts para Resolução de Problemas de Física é uma técnica especializada que utiliza modelos de linguagem grande (LLMs) para auxiliar na análise, solução e explicação de problemas complexos em diversas áreas da física, como mecânica, termodinâmica, eletromagnetismo e física quântica. Envolve a estruturação cuidadosa do *prompt* para fornecer contexto, definir o papel do modelo (ex: tutor, pesquisador, solucionador de problemas), especificar o formato de saída (ex: passo a passo, análise conceitual, código de simulação) e incorporar técnicas avançadas como *Chain-of-Thought* (CoT) ou *Tree-of-Thought* (ToT) para melhorar a precisão e o raciocínio. O objetivo é transformar o LLM de um simples gerador de texto em uma ferramenta de raciocínio técnico e científico, superando as limitações de modelos que tendem a "alucinar" ou pular etapas lógicas em cálculos complexos. Pesquisas recentes (2023-2025) indicam que a combinação de *prompting* estruturado com *Reinforcement Learning with Human-AI Feedback* (RLHF/RLAIF) é a chave para aprimorar o desempenho dos LLMs neste domínio [1].

## Examples
```
**1. Resolução Estruturada (CoT):**
"Você é um físico. Resolva o seguinte problema de cinemática usando o método *Chain-of-Thought* (CoT). Forneça a resposta final em metros por segundo.
Problema: Um carro acelera de 10 m/s para 30 m/s em 5 segundos. Qual é a aceleração média e a distância percorrida?
Passos: 1. Variáveis conhecidas/desconhecidas. 2. Fórmulas. 3. Cálculo da aceleração. 4. Cálculo da distância. 5. Resposta final."

**2. Análise Conceitual e Comparativa:**
"Explique o conceito de dualidade onda-partícula para um aluno do ensino médio. Em seguida, compare as abordagens de Bohr e de Broglie, destacando as diferenças conceituais. Use analogias do cotidiano para facilitar a compreensão."

**3. Simulação e Geração de Código:**
"Gere um código Python (usando a biblioteca `numpy` ou `scipy`) para simular o movimento de um projétil lançado a 45 graus com uma velocidade inicial de 20 m/s. O código deve calcular e plotar a trajetória (posição x tempo) e determinar o alcance máximo. Não inclua explicações no código, apenas o código funcional."

**4. Revisão e Crítica de Solução:**
"Analise a seguinte solução para um problema de circuito RC e identifique se há erros conceituais ou de cálculo. Se houver, corrija-os e forneça a solução correta passo a passo.
Solução Incorreta: [Insira aqui uma solução com erro, ex: esquecendo de converter unidades ou usando a fórmula errada]."

**5. Planejamento de Experimento:**
"Proponha um *setup* experimental detalhado para demonstrar a Lei de Indução de Faraday em um laboratório de física de graduação. O *prompt* deve incluir: 1. Lista de materiais. 2. Procedimento passo a passo. 3. Variáveis a serem medidas. 4. Gráfico esperado dos resultados."

**6. Derivação de Fórmulas:**
"Derive a equação de Bernoulli a partir dos princípios de conservação de energia e do trabalho-energia. Apresente a derivação em formato LaTeX para fácil visualização e inclua uma breve explicação de cada etapa da simplificação."

**7. Resolução Simbólica:**
"Resolva o seguinte problema de colisão inelástica em uma dimensão. Forneça a velocidade final do sistema em termos das massas ($m_1$, $m_2$) e das velocidades iniciais ($v_{1i}$, $v_{2i}$). Não use valores numéricos, apenas manipulação simbólica."
```

## Best Practices
As melhores práticas para a criação de *prompts* eficazes na resolução de problemas de física focam em maximizar a precisão do raciocínio e a clareza da solução.

1.  **Definir o Papel e o Nível de Conhecimento:** Comece o *prompt* instruindo o LLM a assumir um papel específico (ex: "Você é um professor de física de nível universitário") e a adaptar a linguagem e a profundidade da explicação ao público-alvo (ex: "Explique como se fosse para um aluno do ensino médio").
2.  **Estrutura de Resolução de Problemas (CoT/ToT):** Exija que o modelo siga uma estrutura lógica e sequencial. O uso de técnicas como *Chain-of-Thought* (CoT) ou *Tree-of-Thought* (ToT) é crucial. Peça explicitamente: "Primeiro, liste as variáveis conhecidas e desconhecidas. Segundo, declare o princípio físico relevante. Terceiro, apresente a fórmula. Quarto, substitua os valores e calcule. Quinto, forneça a resposta final com unidades."
3.  **Especificar Unidades e Formato:** Sempre inclua as unidades de medida desejadas na resposta (ex: "Forneça a resposta em Newtons e a energia em Joules"). Se necessário, peça o resultado em um formato específico, como código Python para simulação ou uma tabela de dados.
4.  **Fornecer Contexto e Dados:** Inclua todos os dados numéricos, constantes e quaisquer restrições do problema. Para problemas complexos, forneça exemplos de problemas resolvidos (*Few-Shot Prompting*) ou referencie documentos internos (se o LLM tiver essa capacidade).
5.  **Validação e Crítica:** Peça ao LLM para validar sua própria resposta. Por exemplo: "Após resolver, revise a solução e identifique um erro comum que um aluno poderia cometer ao tentar resolver este problema."

## Use Cases
Os *Physics Problem Solving Prompts* são aplicáveis em diversos cenários educacionais, de pesquisa e de desenvolvimento:

*   **Educação e Tutoria:** Geração de soluções passo a passo para problemas de lição de casa, criação de planos de aula detalhados, e desenvolvimento de explicações conceituais adaptadas a diferentes níveis de aprendizado (do ensino fundamental à pós-graduação).
*   **Pesquisa e Desenvolvimento (P&D):** Auxílio na revisão de literatura, sumarização de artigos científicos complexos (ex: sobre física computacional ou astrofísica), e na identificação de lacunas de conhecimento em um campo específico.
*   **Desenho Experimental:** Criação de listas de materiais, procedimentos de laboratório e *checklists* de validação de dados para experimentos de física, garantindo a conformidade com padrões educacionais e de segurança.
*   **Avaliação e Criação de Conteúdo:** Geração de questões de múltipla escolha ou dissertativas para exames, e análise de respostas de alunos para identificar padrões de erros e conceitos mal compreendidos.
*   **Simulação e Modelagem:** Geração de *scripts* de código (Python, MATLAB, etc.) para modelar fenômenos físicos, como movimento de corpos, dinâmica de fluidos ou simulações quânticas, acelerando o processo de prototipagem.

## Pitfalls
Os erros comuns na utilização de LLMs para problemas de física geralmente decorrem da falta de rigor e da confiança excessiva na capacidade de cálculo do modelo.

*   **Alucinação de Fórmulas e Constantes:** O LLM pode citar fórmulas ou valores de constantes físicas incorretos ou inexistentes. **Armadilha:** Não verificar as fórmulas citadas.
*   **Erros de Unidade e Conversão:** O modelo pode misturar unidades (ex: usar centímetros em vez de metros) ou falhar em converter unidades de forma consistente ao longo do cálculo. **Armadilha:** Não especificar as unidades de entrada e saída no *prompt*.
*   **Pular Etapas Lógicas (Raciocínio Superficial):** Em vez de resolver o problema, o LLM pode fornecer a resposta final correta (ou incorreta) sem o processo de raciocínio, o que é inútil para fins didáticos. **Armadilha:** Não exigir explicitamente o método *Chain-of-Thought* (CoT).
*   **Interpretação Incorreta do Contexto:** O modelo pode falhar em interpretar corretamente as condições de contorno ou as restrições físicas implícitas no problema (ex: atrito zero, colisão elástica vs. inelástica). **Armadilha:** Não fornecer um contexto físico completo e claro.
*   **Limitações em Cálculos Simbólicos Complexos:** Embora os LLMs sejam bons em álgebra, problemas de cálculo vetorial ou equações diferenciais parciais complexas podem levar a erros de manipulação simbólica. **Armadilha:** Confiar cegamente em derivações longas sem validação.

## URL
[https://clickup.com/p/ai-prompts/physics-problem-solving](https://clickup.com/p/ai-prompts/physics-problem-solving)
