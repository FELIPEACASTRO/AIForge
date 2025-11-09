# Prompts de Instrução Diferenciada (Differentiated Instruction Prompts)

## Description
**Prompts de Instrução Diferenciada** são uma técnica de Engenharia de Prompt focada em adaptar a saída de um Modelo de Linguagem Grande (LLM) para atender às necessidades de aprendizagem variadas de um público-alvo, tipicamente em um contexto educacional. O objetivo é personalizar o conteúdo, o processo, o produto ou o ambiente de aprendizagem, alinhando-se aos princípios da Instrução Diferenciada de Carol Ann Tomlinson.

A técnica envolve a inclusão de parâmetros explícitos no prompt que definem o **nível de complexidade** (ex: nível de leitura, vocabulário), o **formato de entrega** (ex: artigo informativo, poema, quiz, organizador gráfico), o **interesse do aluno** (ex: tema de *Fortnite*, *Minecraft*, esportes) e a **habilidade cognitiva** a ser trabalhada (ex: resumir, analisar, inferir, aplicar).

Ao especificar esses elementos, o usuário (geralmente um educador) transforma a IA em uma ferramenta poderosa para gerar materiais de ensino e avaliação sob medida, economizando tempo e garantindo que cada aluno receba um desafio apropriado e relevante para seu perfil de aprendizagem [1] [2].

## Examples
```
1. **Diferenciação por Nível de Leitura e Interesse:**
   `Aja como um tutor de história. Crie um artigo informativo de 500 palavras sobre a Revolução Francesa. O artigo deve ser escrito em um nível de leitura da 5ª série, mas com um tom envolvente para um aluno de 9º ano que adora jogos de estratégia. Inclua uma seção que compare a logística da Revolução com a de um jogo de estratégia.`

2. **Diferenciação por Habilidade Cognitiva e Formato:**
   `Você é um assistente de professor de ciências. Gere um conjunto de 3 perguntas de resposta curta sobre o ciclo da água. As perguntas devem focar na habilidade de **análise** (não apenas memorização) e devem ser formatadas como um desafio para um aluno que já domina o conceito básico.`

3. **Diferenciação por Produto (Saída) e Nível:**
   `Para um aluno com TDAH que precisa de estrutura visual, crie um organizador gráfico vazio (apenas o texto da estrutura) para ajudá-lo a planejar uma redação argumentativa sobre a importância da reciclagem. O organizador deve ter apenas 3 seções principais e usar linguagem simples e direta.`

4. **Diferenciação por Processo (Instrução):**
   `Crie um plano de aula de 45 minutos para ensinar a habilidade de **inferência** em um texto narrativo. O plano deve incluir uma atividade de modelagem (eu faço), uma atividade guiada (nós fazemos) e uma atividade independente (você faz). O texto narrativo deve ser sobre um evento esportivo.`

5. **Diferenciação para Alunos Multilíngues (ELL/ESL):**
   `Traduza o seguinte parágrafo sobre fotossíntese para o português e, em seguida, crie uma lista de 5 palavras-chave em inglês com suas definições simplificadas em português. [PARÁGRAFO AQUI]`

6. **Diferenciação para Enriquecimento (Avançado):**
   `Aja como um professor universitário. Crie um prompt de pesquisa para um aluno avançado que já domina o conceito de gravidade. O prompt deve exigir que o aluno explore a relação entre a teoria da relatividade de Einstein e a mecânica quântica, e deve ser formatado como uma proposta de ensaio de 1500 palavras.`

7. **Diferenciação de Material (Múltiplos Níveis):**
   `Gere três versões do mesmo texto informativo sobre a estrutura de uma célula:
   a) Versão A: Nível de leitura da 3ª série, com analogias simples (ex: célula como uma casa).
   b) Versão B: Nível de leitura da 7ª série, com vocabulário científico padrão.
   c) Versão C: Nível de leitura da 11ª série, incluindo detalhes sobre organelas e suas funções bioquímicas.`
```

## Best Practices
**Especificidade e Contexto:** Sempre inclua o papel do usuário (ex: "Professor de História do 9º ano"), o público-alvo (ex: "Alunos com nível de leitura do 6º ano"), o formato de saída (ex: "Um quiz de 5 perguntas de múltipla escolha"), e o foco instrucional (ex: "Foco na habilidade de inferência"). **Modularidade:** Crie prompts que possam ser facilmente adaptados para diferentes níveis, interesses ou habilidades, alterando apenas um ou dois parâmetros (ex: mudar o nível de leitura de "4ª série" para "8ª série"). **Verificação Humana:** Nunca use o conteúdo gerado pela IA sem uma revisão cuidadosa para garantir a precisão factual, o nível de leitura apropriado e a sensibilidade cultural. **Integração Pedagógica:** Use a IA como um assistente para a criação de material, mas mantenha o papel central do professor no fornecimento de *scaffolding* (apoio), *feedback* e conexão humana.

## Use Cases
**Criação de Material Didático Personalizado:** Geração rápida de textos, exercícios, *quizzes* e planos de aula adaptados para diferentes níveis de leitura, estilos de aprendizagem (visual, auditivo, cinestésico) e interesses temáticos dos alunos. **Apoio a Necessidades Específicas:** Criação de materiais com acomodações para alunos com dificuldades de aprendizagem, TDAH, ou alunos de inglês como segunda língua (ELL/ESL), ajustando a complexidade da linguagem e o formato de apresentação [1]. **Avaliação Formativa Adaptativa:** Geração de perguntas de avaliação que se ajustam em tempo real ao progresso do aluno, focando em habilidades específicas que precisam de reforço (ex: gerar mais perguntas sobre "causa e efeito" para um aluno que demonstrou dificuldade nessa área). **Enriquecimento e Aceleração:** Criação de projetos de pesquisa e materiais de aprofundamento para alunos avançados, permitindo que explorem o conteúdo em um nível de complexidade superior.

## Pitfalls
**Falta de Verificação:** Confiar cegamente no nível de leitura ou precisão factual gerada pela IA. O nível de leitura especificado pode ser impreciso, exigindo ferramentas de verificação de legibilidade e revisão humana [2]. **Substituição do Professor:** Usar a IA para substituir o planejamento pedagógico e o *scaffolding* (apoio) humano. A IA é uma ferramenta de criação de material, não um substituto para a expertise e a conexão do professor [2]. **Prompts Genéricos:** Usar prompts vagos que não especificam o público, o nível ou a habilidade. Isso resulta em conteúdo não diferenciado e de baixa qualidade. **Viés e Inadequação:** A IA pode gerar conteúdo com viés ou inadequado para a idade/cultura do aluno. A revisão de sensibilidade é crucial [2].

## URL
[https://schoolai.com/blog/strategies-using-ai-tutors-improve-differentiated-instruction/](https://schoolai.com/blog/strategies-using-ai-tutors-improve-differentiated-instruction/)
