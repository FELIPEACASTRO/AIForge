# Formulação de Perguntas de Pesquisa (Research Question Formulation)

## Description
A Formulação de Perguntas de Pesquisa, no contexto da Engenharia de Prompt, é a **técnica fundamental de estruturar e refinar consultas (prompts) para modelos de Linguagem Grande (LLMs) de forma a obter respostas que não apenas sejam corretas, mas que também avancem o conhecimento ou resolvam um problema específico, simulando o rigor de uma investigação científica ou de negócios.** É a arte de fazer as "perguntas certas" para a IA, transformando uma necessidade vaga em uma instrução específica, bem direcionada e rica em contexto. O foco é na **qualidade da entrada (input)**, reconhecendo que a precisão e a utilidade da saída (output) do LLM são diretamente proporcionais à clareza e profundidade da pergunta formulada. Esta técnica se alinha com a filosofia de que o foco principal não é a "engenharia de prompt" em si, mas sim a **formulação do problema** ou da pergunta de pesquisa.

## Examples
```
1. **Acadêmico (Refinamento de Tese):**
```
**Contexto:** Sou um pesquisador de doutorado em Ciência da Computação. Meu tópico inicial é "O Impacto da IA Generativa na Produtividade de Desenvolvedores de Software".
**Instrução:** Refine este tópico em 3 perguntas de pesquisa distintas e testáveis, cada uma focada em um aspecto diferente (eficiência de código, satisfação do desenvolvedor e custo-benefício). Para cada pergunta, sugira uma metodologia de pesquisa (ex: estudo de caso, experimento controlado, survey).
**Formato:** Lista numerada com a Pergunta de Pesquisa em negrito, seguida pela Metodologia Sugerida.
```

2. **Desenvolvimento de Produto (User Story):**
```
**Contexto:** Somos uma equipe de desenvolvimento de um aplicativo de gestão financeira. Identificamos que muitos usuários abandonam o cadastro na etapa de "Conexão Bancária".
**Instrução:** Formule a principal Pergunta de Pesquisa que devemos responder para resolver este problema. Em seguida, crie uma User Story (História de Usuário) completa, seguindo o formato "Como um [Tipo de Usuário], eu quero [Objetivo], para que [Benefício]".
**Restrição:** A User Story deve focar na redução da fricção e aumento da confiança.
```

3. **Estratégia de Negócios (Análise de Mercado):**
```
**Contexto:** Nossa empresa de SaaS B2B está considerando expandir para o mercado europeu. O produto é uma ferramenta de automação de marketing para pequenas e médias empresas (PMEs).
**Instrução:** Formule a Pergunta de Pesquisa Estratégica mais crítica que precisamos responder antes de alocar recursos significativos. Em seguida, liste 5 sub-perguntas táticas que a IA deve responder para apoiar a resposta à pergunta principal.
```

4. **Diagnóstico e Solução de Problemas (Troubleshooting - Meta-Prompting):**
```
**Instrução:** Estou enfrentando um problema persistente de latência em meu banco de dados PostgreSQL após a última atualização de software. Em vez de me dar uma solução direta, aja como um Engenheiro de Sistemas Sênior. Faça-me 5 perguntas de diagnóstico cruciais sobre meu ambiente e configuração (versão do S.O., tipo de hardware, logs de erro, etc.) que você precisaria para começar a formular uma hipótese de causa raiz.
```

5. **Geração de Hipótese Científica:**
```
**Contexto:** Observamos que a taxa de cliques (CTR) em nossos anúncios de mídia social é 30% maior em imagens que contêm a cor azul em comparação com outras cores.
**Instrução:** Formule uma hipótese nula (H0) e uma hipótese alternativa (H1) testáveis para um experimento A/B que visa confirmar ou refutar essa observação.
**Formato:** H0: [Hipótese Nula] e H1: [Hipótese Alternativa].
```

6. **Reflexão e Auto-Correção (Prompt de Refinamento):**
```
**Instrução:** Analise o prompt que acabei de usar: "Escreva um e-mail de marketing". Identifique as 3 principais deficiências deste prompt em termos de contexto, instrução e formato. Em seguida, reescreva o prompt para torná-lo um "prompt de pesquisa" avançado para a criação de um e-mail de marketing de alta conversão.
```
```

## Best Practices
**Seja Específico e Detalhado:** Forneça o máximo de contexto, restrições e intenção possível. A clareza na entrada é o fator mais crítico para a qualidade da saída. **Quebre a Pergunta:** Divida tarefas complexas em uma série de perguntas menores e sequenciais. **Peça o Raciocínio (Chain-of-Thought):** Solicite que a IA explique a lógica por trás de suas sugestões. Isso permite a verificação e o refinamento progressivo. **Defina o Formato:** Especifique o formato de saída desejado (tabela, lista, código, etc.) e o tom (formal, didático, técnico). **Iteração é Chave:** Use a resposta da IA para refinar e aprofundar a próxima pergunta, em um ciclo contínuo de investigação.

## Use Cases
**Pesquisa Acadêmica:** Geração de hipóteses, refinamento de perguntas de pesquisa para teses e artigos, e planejamento de revisões sistemáticas de literatura. **Desenvolvimento de Produto/Software:** Definição de requisitos de usuário (User Stories), priorização de *features* em *roadmaps*, e análise de UI/UX. **Consultoria e Estratégia de Negócios:** Análise de cenários de mercado, formulação de estratégias de entrada em novos mercados, e identificação de riscos e oportunidades. **Resolução de Problemas Complexos (Troubleshooting):** Diagnóstico de problemas persistentes em sistemas, solicitando à IA que faça perguntas de diagnóstico para entender o contexto. **Criação de Conteúdo Didático:** Elaboração de planos de aula e criação de questões de múltipla escolha ou abertas com base em um texto.

## Pitfalls
**Ambiguidade e Generalização Excessiva:** Usar prompts muito amplos ("Me ajude a melhorar meus processos") ou ambíguos que permitem múltiplas interpretações. **Falta de Contexto:** Não fornecer o cenário, o papel da IA ou os dados de entrada necessários para que a IA entenda a profundidade da questão. **Perguntas Múltiplas/Compostas:** Tentar resolver vários problemas em uma única consulta, resultando em respostas superficiais e incompletas. **Vieses Inconscientes:** Formular a pergunta de forma a induzir a IA a uma resposta pré-determinada, limitando a criatividade e a análise crítica. **Ignorar a Iteração:** Tratar o prompt como uma única interação em vez de um processo de refinamento progressivo.

## URL
[https://medium.com/@petrusje/engenharia-de-prompts-a-arte-de-fazer-perguntas-certas-para-ia-14f9e5c57045](https://medium.com/@petrusje/engenharia-de-prompts-a-arte-de-fazer-perguntas-certas-para-ia-14f9e5c57045)
