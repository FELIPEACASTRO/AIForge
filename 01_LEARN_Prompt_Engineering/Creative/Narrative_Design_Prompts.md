# Narrative Design Prompts

## Description
A técnica de **Narrative Design Prompts** (Prompts de Design Narrativo) é uma abordagem de Engenharia de Prompt focada em utilizar Modelos de Linguagem Grande (LLMs) para auxiliar na criação, estruturação e desenvolvimento de narrativas complexas, especialmente em contextos de jogos, roteiros, UX/UI (User Experience/User Interface) e storytelling interativo. Em vez de apenas gerar texto, o prompt é desenhado para atuar como um **co-designer narrativo**, solicitando à IA que aplique estruturas narrativas conhecidas (como a Jornada do Herói ou o Save the Cat!), desenvolva perfis psicológicos de personagens, crie árvores de diálogo ramificadas ou sugira elementos de *worldbuilding* (construção de mundo). O objetivo é automatizar o "trabalho braçal" da criação de conteúdo incidental (como diálogos de NPCs) ou acelerar o *brainstorming* de estruturas complexas, permitindo que o designer se concentre na visão artística e na coesão geral da experiência narrativa. Esta técnica é fundamental para a **democratização do design narrativo** em projetos de pequena e média escala [1] [2].

## Examples
```
**1. Criação de Beat Sheet (Estrutura de Batidas):**
`Aja como Blake Snyder, aplicando a estrutura "Save the Cat!". Minha logline é: "Um ex-chef de cozinha, agora um caçador de recompensas intergaláctico, deve encontrar um ingrediente raro em um planeta hostil para salvar o paladar de sua filha, que está morrendo." Gere um "beat sheet" completo com os 15 pontos, descrevendo brevemente o que acontece em cada um.`

**2. Desenvolvimento de Personagem (Psicologia):**
`Aja como um psicólogo de personagens. Crie um perfil psicológico para o protagonista de um RPG de fantasia sombria. Nome: Kael. Objetivo consciente: Vingar a morte de seu irmão. Medo inconsciente: Ser a causa da próxima tragédia. Necessidade: Aprender a confiar em um grupo. Descreva a contradição central de seu caráter e como isso se manifesta em seu diálogo.`

**3. Geração de Diálogo Ramificado (Árvore de Diálogo):**
`Gere uma árvore de diálogo para um NPC (Guarda da Cidade) em um jogo de aventura. O jogador precisa convencê-lo a abrir o portão. O guarda é cético e leal. Forneça 3 opções de diálogo para o jogador (Persuasão, Suborno, Ameaça) e as 3 respostas correspondentes do NPC, incluindo o resultado de cada caminho (Sucesso/Falha).`

**4. Worldbuilding (Construção de Mundo) e Conflito:**
`Aja como um designer de mundo. Meu cenário é uma metrópole futurista chamada "Neo-Veridia", onde a água potável é o recurso mais escasso. Gere 5 facções sociais distintas que lutam pelo controle da água, descrevendo o nome, a ideologia e o principal conflito narrativo de cada uma.`

**5. Design de Missão (Quest Design):**
`Crie uma missão secundária (side quest) para um jogo de ficção científica. O tema da missão deve ser "O peso da memória". O jogador deve recuperar um artefato. Descreva o Ponto de Partida, o Objetivo, o Conflito Moral (a escolha difícil que o jogador deve fazer) e a Recompensa (que deve ser mais emocional do que material).`

**6. Sugestão de Elementos de Gamificação (UX Narrativa):**
`Aja como um designer de UX/UI. Estamos gamificando um aplicativo de aprendizado de idiomas. O tema narrativo é "Jornada de um Explorador Linguístico". Sugira 3 elementos de gamificação (ex: medalhas, barras de progresso, pontos) e descreva como a narrativa do explorador se conecta a cada um deles.`
```

## Best Practices
**Definir o Papel da IA (Persona):** Comece o prompt instruindo a IA a assumir um papel específico, como "Designer Narrativo Sênior", "Roteirista de Cinema" ou "Psicólogo de Personagens". Isso alinha o tom e a perspectiva da resposta. **Usar Estruturas Narrativas Conhecidas:** Incorpore explicitamente modelos narrativos (ex: Jornada do Herói, Estrutura de Três Atos, Save the Cat!) para guiar a IA na organização da história. **Fornecer Contexto Detalhado:** Quanto mais detalhes sobre o mundo, personagens e premissa, mais rica e coesa será a saída. Inclua gênero, tom, público-alvo e restrições. **Focar em Elementos Específicos:** Em vez de pedir a história inteira, peça elementos modulares: a curva de transformação do vilão, 5 diálogos de NPC, a lógica de uma *quest* secundária. **Iterar e Refinar:** Use a saída da IA como um rascunho. Peça refinamentos específicos, como "Reescreva o Ato II, aumentando o senso de urgência e adicionando um novo mentor."

## Use Cases
**Desenvolvimento de Jogos:** Criação rápida de diálogos de NPCs (personagens não jogáveis), design de missões secundárias, estruturação de arcos de história ramificados e validação de lógicas de *worldbuilding* [1]. **Roteiro e Cinema:** Geração de *beat sheets* (folhas de batidas), *loglines*, sinopses e perfis psicológicos de personagens para acelerar a fase de pré-produção [2]. **Storytelling Interativo:** Criação de narrativas dinâmicas que se adaptam às escolhas do usuário em tempo real, como em experiências de realidade virtual ou livros-jogos digitais. **UX/UI e Gamificação:** Aplicação de estruturas narrativas (ex: Jornada do Herói) para desenhar a experiência do usuário em aplicativos, tornando o uso mais envolvente e motivador. **Educação e Treinamento:** Criação de cenários de simulação baseados em histórias para treinamento corporativo ou militar, onde a IA gera as respostas e consequências dos personagens.

## Pitfalls
**Geração de Conteúdo Genérico:** A IA pode gerar narrativas previsíveis ou clichês se o prompt for muito vago. É crucial fornecer detalhes e restrições específicas. **Perda de Coerência em Longo Prazo:** LLMs podem ter dificuldade em manter a consistência de detalhes complexos (como regras de *worldbuilding* ou traços de personalidade de personagens) ao longo de múltiplas interações. **Dependência Excessiva:** Usar a IA para criar a narrativa completa pode levar à perda da voz autoral e da profundidade criativa. A IA deve ser uma ferramenta de *brainstorming* e rascunho, não o autor final. **Foco na Forma, Não na Função:** O prompt pode gerar uma estrutura narrativa perfeita (ex: 15 *beats* do Save the Cat!), mas sem o conteúdo emocional ou temático necessário para torná-la envolvente. O designer deve sempre injetar o "coração" da história.

## URL
[https://yenra.com/ai20/interactive-storytelling-and-narratives//](https://yenra.com/ai20/interactive-storytelling-and-narratives//)
