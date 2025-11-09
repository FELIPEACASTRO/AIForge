# Prompt Engineering para Escrita Criativa (Ficção e Poesia)

## Description

A Engenharia de Prompts para Escrita Criativa (Ficção e Poesia) foca em técnicas para guiar Modelos de Linguagem Grandes (LLMs) a produzir textos narrativos e poéticos de alta qualidade, criativos e coerentes. As técnicas mais eficazes envolvem a definição de um **Estilo de Escrita de Referência** (transferência de estilo), a aplicação de **Restrições Estruturais** (como gênero, tom, ponto de vista e formato) e o uso de **Co-criação Iterativa** (refinando o texto em etapas). Pesquisas recentes (2024-2025) indicam que, embora os LLMs gerem textos linguisticamente complexos, eles tendem a ter **menor novidade, diversidade e surpresa** em comparação com escritores humanos, sugerindo que a criatividade artificial ainda é limitada pela repetição de padrões de treinamento. A melhor prática é usar o LLM como um **co-piloto criativo**, fornecendo contexto detalhado (personagens, cenário, enredo) e solicitando *outlines* ou rascunhos de capítulos/estrofes, em vez de pedir a história completa de uma só vez.

## Statistics

Um estudo de 2025 (arXiv:2411.02316v4) comparando 60 LLMs e 60 humanos na geração de histórias curtas criativas (baseadas em 3 palavras-chave) revelou que:
- **Complexidade Linguística:** LLMs geram histórias com estruturas linguisticamente mais complexas do que humanos.
- **Criatividade:** LLMs e não-especialistas tendem a classificar histórias geradas por LLMs como mais criativas do que as humanas.
- **Novidade, Diversidade e Surpresa:** As histórias de LLMs **significativamente ficam aquém** dos escritores humanos médios em termos de novidade, diversidade e surpresa.
- **Avaliação de Especialistas:** Julgamentos de especialistas correlacionam-se positivamente com métricas automatizadas, focando na complexidade semântica em vez da complexidade linguística.
- **Co-criação:** Pesquisas (ACM 2025) indicam que escritores profissionais usam LLMs em todas as etapas do processo criativo, principalmente para *brainstorming* e geração de rascunhos.

## Features

**Transferência de Estilo (Style Transfer):** Análise de um texto de exemplo para extrair e replicar o estilo, tom e temas do autor. **Prompting Iterativo e Co-criação:** Uso de *outlines* (esboços) e refinamento em múltiplas etapas para manter a coerência narrativa e o desenvolvimento de personagens. **Definição de Restrições:** Especificação de gênero (ex: *gothic-punk*), forma (ex: soneto, poesia concreta), ponto de vista (ex: primeira pessoa limitada) e *show, don't tell* para melhorar a qualidade da prosa. **Uso de Palavras-Chave (Cue Words):** Fornecimento de palavras-chave para ancorar a criatividade e o tema central do texto.

## Use Cases

**Geração de Rascunhos:** Criação rápida de primeiros rascunhos de capítulos, cenas ou estrofes. **Transferência de Estilo:** Emulação do estilo de um autor específico ou de um gênero literário (ex: *noir*, realismo mágico). **Desenvolvimento de Personagens:** Exploração de *backstories*, diálogos e arcos de personagens complexos. **Poesia Estruturada:** Geração de poemas com formas específicas (sonetos, haicais, poesia concreta) onde a estrutura é uma restrição de prompt. **Bloqueio Criativo:** Superação do bloqueio criativo através da geração de ideias de enredo ou reviravoltas inesperadas.

## Integration

**Exemplo de Prompt para Transferência de Estilo (Ficção):**
"Analise o texto a seguir para seu estilo de escrita, tom, temas e outros aspectos literários. Forneça um guia abrangente baseado nessas características que possa ser usado como referência para escrever uma história original emulando o estilo distinto deste autor. O guia deve ser abstrato e abrangente o suficiente para permitir a criação de novas ideias originais que pareçam semelhantes ao original. Não cite nomes de personagens; refira-se a eles como Personagem1, Personagem2, etc. [TEXTO DE EXEMPLO]"

**Exemplo de Prompt para Poesia Concreta:**
"Gere um poema concreto sobre o tema 'Caos Urbano' usando a forma visual de um arranha-céu em colapso. O poema deve usar apenas as palavras 'concreto', 'aço', 'grito', 'silêncio' e 'poeira'. O layout visual é tão importante quanto as palavras."

**Melhores Práticas:**
1. **Defina o Papel:** Comece com `Você é um [Gênero/Estilo] autor renomado...`
2. **Contexto Detalhado:** Forneça detalhes ricos sobre personagens (objetivos, medos), cenário (sensorial) e conflito.
3. **Estrutura:** Peça um *outline* de capítulos/estrofes antes de solicitar o texto completo.
4. **Iteração:** Peça o texto em pequenos blocos (ex: "Escreva o Capítulo 1, focando na tensão entre Personagem A e B").
5. **Revisão de Estilo:** Use comandos como `Reescreva o parágrafo anterior usando a técnica 'show, don't tell'`.

## URL

https://medium.com/@daniellefranca96/how-to-write-creative-fictional-texts-with-llms-38720d119efd