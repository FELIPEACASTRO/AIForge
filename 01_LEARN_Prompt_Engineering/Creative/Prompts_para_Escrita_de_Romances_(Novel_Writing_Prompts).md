# Prompts para Escrita de Romances (Novel Writing Prompts)

## Description
A técnica de **Prompts para Escrita de Romances** é uma aplicação especializada da Engenharia de Prompt focada em utilizar Modelos de Linguagem Grande (LLMs) para auxiliar em todas as fases da criação de um romance, desde o *brainstorming* inicial até a edição e revisão de capítulos. Em vez de pedir à IA para escrever o livro inteiro de uma só vez, o método envolve a criação de prompts estruturados e interativos que guiam a IA na construção de elementos complexos como *world-building*, perfis detalhados de personagens, arcos de enredo, glossários e apêndices. O objetivo é transformar a IA em um co-piloto de escrita, que mantém a consistência narrativa e aprofunda o material de referência do projeto.

## Examples
```
### Exemplos de Prompts (5-10 Exemplos Concretos)

**1. Prompt de Configuração Inicial (Master Prompt):**
```
# Papel e Contexto
Você é um escritor e editor de fantasia sombria (Grimdark Fantasy) especialista, com atenção meticulosa aos detalhes. Seu objetivo é auxiliar na criação de um romance de alta qualidade e bem estruturado.

## Estilo de Escrita
- Mantenha um tom cínico e pessimista.
- Use prosa descritiva densa e diálogos concisos.
- Perspectiva em terceira pessoa, passado.
- Evite clichês como "mergulhar em" ou "liberar seu potencial".

## Processo
- Revise todos os arquivos de conhecimento do projeto antes de responder.
- Mantenha a consistência narrativa com o "World-Building Framework" e os "Character Profiles".
- Conclua cada interação com uma pergunta que me ajude a avançar no romance.
```

**2. Prompt de Criação de Documentos de Referência:**
```
* TÍTULO: O Último Suspiro do Sol
* Gênero: Fantasia Sombria Pós-Apocalíptica
* Premissa: Em um mundo onde o sol morreu há um século, um ex-paladino viciado em drogas deve escoltar uma criança com a chave para reacender a luz através de terras infestadas por criaturas da escuridão.
* Protagonista: Kael, um paladino caído, cínico, com um código moral quebrado e uma dependência de "Pó de Estrela" (uma droga local).

Crie os seguintes documentos para este romance, detalhando-os extensivamente:
1. World-Building Framework (Cosmologia, Geografia, Sistema de Magia, Sociedade).
2. Character Profile: Kael (História, Motivações, Arco de Personagem).
3. Supporting Cast Profiles (Aliados e Antagonistas Principais).
4. Plot Outline (Esboço Capítulo por Capítulo, Arcos de Personagem, Temas).
```

**3. Prompt de Geração de Capítulo:**
```
Com base no "Plot Outline" e usando o "World-Building Framework" e o "Character Profile: Kael" como contexto, elabore o Rascunho do Capítulo 3.

Antes de escrever, crie um plano de execução detalhado para o capítulo, incluindo:
- Objetivo do Capítulo (O que deve ser alcançado).
- Pontos de Enredo Chave.
- Conflito (Interno e Externo).
- Palavras-chave de Vocabulário (para consistência).
- Contagem de Palavras Estimada (mínimo 2500 palavras).

O capítulo deve terminar com Kael tomando uma decisão arriscada que o coloca em perigo imediato.
```

**4. Prompt de Atualização de Documentos de Referência:**
```
O Capítulo 3 foi concluído. Por favor, atualize os seguintes documentos para incorporar todos os novos termos, locais e eventos introduzidos no Capítulo 3:
1. Plot Outline (Marque o Capítulo 3 como concluído e revise a progressão do Capítulo 4).
2. Glossary (Adicione novos termos específicos do mundo, como "Pó de Estrela" e "Os Sem-Luz").
3. Index (Adicione referências de capítulo para novos personagens secundários e locais).
```

**5. Prompt de Crítica e Edição (Desenvolvimento):**
```
Analise o texto do Capítulo 5 (anexado) e forneça uma crítica de desenvolvimento.

Foque nos seguintes aspectos:
- **Ritmo:** Onde a ação desacelera ou acelera demais?
- **Continuidade:** Há alguma inconsistência com o "World-Building Framework" ou "Character Profile" de Kael?
- **Diálogo:** O diálogo entre Kael e o Paladino Caído soa autêntico e avança o enredo?

Forneça um relatório detalhado com sugestões de revisão concretas.
```

**6. Prompt de Expansão de Cena:**
```
A cena atual (linhas 15-30 do Capítulo 7) é muito curta. Expanda esta cena para focar na descrição sensorial do ambiente: o cheiro de ozônio e metal queimado, o som do vento uivando pelas ruínas e a sensação de areia fina sob as botas de Kael. Aumente a contagem de palavras em pelo menos 500.
```

**7. Prompt de Geração de Diálogo:**
```
Crie um diálogo tenso entre o protagonista (Kael) e o principal antagonista (A Sombra).

**Contexto:** Kael foi capturado. A Sombra está tentando convencê-lo a se juntar à causa dela, explorando o código moral quebrado de Kael e sua dependência de drogas.
**Objetivo:** O diálogo deve revelar uma fraqueza inesperada na Sombra e um momento de hesitação em Kael.
```

**8. Prompt de Brainstorming de Título/Capa:**
```
Gere 10 títulos alternativos para o romance "O Último Suspiro do Sol" que sejam mais sombrios e evoquem a sensação de desespero e fantasia sombria. Além disso, descreva em detalhes uma imagem de capa que capture a essência do gênero e da premissa.
```
```

## Best Practices
1.  **Estabelecer um "Master Prompt" (Instrução Mestra):** Definir um prompt inicial que estabelece o papel da IA (ex: "Escritor e editor de romances especialista"), o contexto e as responsabilidades principais (ex: manter a consistência, produzir conteúdo sem restrições de comprimento).
2.  **Manter uma Base de Conhecimento (Knowledge Base) Atualizada:** Utilizar o sistema de arquivos ou recurso de contexto da plataforma (como o "Project Knowledge" do Claude ou "Codex" do NovelCrafter) para armazenar e atualizar continuamente documentos de referência (Capítulos, Perfis de Personagens, Estrutura do Mundo, Esboço do Enredo, Glossário, Apêndice).
3.  **Estrutura Modular e Iterativa:** Dividir o processo de escrita em módulos (capítulos, cenas, perfis) e usar um novo tópico de conversa para cada capítulo ou cena principal. Isso ajuda a gerenciar a janela de contexto e garante que a IA se concentre apenas nas informações mais relevantes.
4.  **Definir um Estilo de Escrita:** Incluir instruções específicas sobre o tom, ponto de vista (POV), tempo verbal e até mesmo aversões estilísticas (ex: "Evite o uso excessivo de travessões ou ponto e vírgula") para garantir que a prosa gerada se alinhe à voz do autor.
5.  **Revisão e Refinamento Constantes:** Após a geração de cada capítulo ou elemento, usar prompts de edição e crítica (ex: "Critique o ritmo, a estrutura e a continuidade deste capítulo") para refinar o material e garantir a coesão com o restante da obra.

## Use Cases
*   **Brainstorming e Concepção Inicial:** Gerar ideias para títulos, premissas, gêneros e estruturas de alto nível.
*   **Construção de Mundo (World-Building):** Criar detalhes extensivos sobre cosmologia, sistemas de magia, geografia, política e cronogramas históricos.
*   **Desenvolvimento de Personagens:** Elaborar perfis psicológicos, arcos de personagem, motivações e vozes distintas para protagonistas e elenco de apoio.
*   **Esboço Detalhado do Enredo:** Criar um detalhamento capítulo por capítulo, incluindo a progressão do arco do personagem e elementos temáticos.
*   **Manutenção de Consistência:** Usar a IA para atualizar documentos de referência (Glossário, Índice) a cada novo capítulo, garantindo que termos, locais e eventos sejam usados de forma consistente.
*   **Edição e Crítica:** Solicitar à IA uma crítica de desenvolvimento sobre ritmo, estrutura e continuidade de um capítulo já escrito.

## Pitfalls
*   **Inconsistência de Contexto:** Falha em atualizar a base de conhecimento da IA com novos capítulos ou revisões, levando a erros de continuidade (o maior problema em projetos longos).
*   **"Voz" Genérica da IA:** A IA pode cair em clichês ou usar uma linguagem genérica se o prompt não incluir um guia de estilo detalhado ou exemplos de escrita do autor.
*   **Dependência Excessiva:** Confiar na IA para escrever o rascunho completo sem intervenção do autor, resultando em uma história que carece de profundidade emocional e voz autêntica.
*   **Limitação de Contexto (Context Window):** Em plataformas sem um recurso de "projeto" ou "base de conhecimento", o contexto da conversa pode ser perdido, exigindo que o autor insira o "resumo da história até agora" ou o último capítulo a cada novo prompt.
*   **Nomes Repetitivos:** A IA pode reutilizar nomes de personagens ou locais se não for instruída a usar geradores de nomes aleatórios ou se o autor não revisar e substituir os nomes gerados.

## URL
[https://www.reddit.com/r/WritingWithAI/comments/1kje334/aiassisted_novel_writing_guide/](https://www.reddit.com/r/WritingWithAI/comments/1kje334/aiassisted_novel_writing_guide/)
