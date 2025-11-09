# Story Arc Prompts

## Description
Story Arc Prompts é uma técnica de engenharia de prompt que utiliza uma abordagem sistemática e em camadas para guiar Modelos de Linguagem Grande (LLMs) na criação de arcos de personagem ou narrativas complexas e autênticas. Em vez de um único prompt, envolve uma série de prompts estratégicos e encadeados que definem o ponto de partida do personagem (crenças, falhas), o catalisador (incidente incitante), a jornada de transformação (progresso e retrocessos) e a resolução final. O objetivo é evitar transformações superficiais e garantir profundidade, nuance emocional e consistência narrativa. A técnica é crucial para a criação de conteúdo de ficção de alta qualidade, pois força o LLM a simular a complexidade do desenvolvimento humano e narrativo, em vez de entregar um simples instantâneo de "antes e depois".

## Examples
```
**Exemplo 1: Estabelecendo a Fundação (Prompt de Essência)**
"Descreva um personagem chamado Elias, que é profundamente cínico em relação à tecnologia devido a um evento traumático de perda de dados em sua juventude. Mostre como esse cinismo se manifesta em suas interações diárias (evita smartphones, usa apenas papel e caneta) e em seus pensamentos internos (acredita que a IA é inerentemente falha e perigosa). Inclua três comportamentos específicos que demonstrem essa desconfiança."

**Exemplo 2: Construindo o Mundo (Prompt de Relacionamento)**
"Crie um relacionamento chave para Elias com uma colega de trabalho chamada Sofia, que é uma entusiasta e desenvolvedora de IA. Descreva três interações onde a paixão de Sofia pela tecnologia desafia diretamente a visão de mundo cínica de Elias. O relacionamento deve ser de respeito mútuo, mas com tensão constante."

**Exemplo 3: Identificando o Catalisador (Prompt de Incidente Incitante)**
"Qual evento específico e significativo poderia forçar Elias a questionar seu cinismo central em relação à tecnologia? O evento deve ser autêntico ao seu mundo (talvez um desastre natural ou uma crise pessoal) e grande o suficiente para quebrar sua visão de mundo existente. Forneça três possíveis catalisadores, cada um com um impacto emocional diferente."

**Exemplo 4: Mapeando a Mudança (Prompt de Tentativa e Falha)**
"Após o catalisador (escolha um dos três), descreva o primeiro pequeno passo de Elias em direção à mudança. Mostre-o tentando usar uma ferramenta tecnológica simples (como um aplicativo de mapa), falhando miseravelmente e o que ele aprende com o fracasso. Inclua sua resistência interna e o monólogo que o faz quase desistir."

**Exemplo 5: Aprofundando a Mudança (Prompt de Contradição)**
"Dê a Elias um comportamento que contradiz seu cinismo: ele secretamente mantém um blog anônimo sobre a beleza da caligrafia e da arte analógica. Explique a razão psicológica para essa contradição: é uma forma de controle ou uma busca por beleza em um mundo que ele vê como caótico e digital. Use essa contradição para criar um momento de vulnerabilidade."

**Exemplo 6: O Clímax (Prompt de Crise)**
"Crie um ponto de crise onde Elias deve escolher entre seu velho hábito (confiar apenas em métodos analógicos, como um mapa de papel) e seu novo eu emergente (confiar em uma solução tecnológica, como um sistema de IA de navegação) para salvar Sofia de um perigo iminente. Torne as apostas pessoais e a escolha inevitável. Mostre o conflito interno e a decisão final."

**Exemplo 7: Selando a Transformação (Prompt de Resolução)**
"Após a decisão no clímax, como Elias demonstra sua transformação em uma ação que contrasta diretamente com seu comportamento no Exemplo 1? Ele deve usar a tecnologia de forma proativa e confiante. Descreva a cena final onde ele não apenas usa a tecnologia, mas também a defende, mostrando o quão longe ele chegou."
```

## Best Practices
**Estrutura em Fases:** Utilize a estrutura de três fases (Fundação, Jornada, Resolução) para garantir que o LLM construa a narrativa de forma incremental e lógica. **Prompts em Camadas:** Nunca use um único prompt. Em vez disso, use uma série de prompts encadeados, onde a saída de um prompt informa o próximo, forçando o LLM a construir a complexidade gradualmente. **Mapeie o "Meio Bagunçado":** Peça explicitamente por falhas, retrocessos e resistência interna à mudança. A transformação não é linear e o LLM precisa ser instruído a criar o "meio bagunçado" para que o arco seja crível. **Use o Método da Contradição:** Introduza traços ou comportamentos contraditórios no personagem e peça ao LLM para explicar a razão psicológica. Isso adiciona profundidade e evita personagens unidimensionais. **Ancore a Voz:** Mesmo com a transformação, ancore a voz do personagem com um tique verbal ou crença persistente para mantê-lo reconhecível.

## Use Cases
**Criação de Ficção:** Desenvolvimento de personagens complexos e narrativas envolventes para livros, roteiros, peças de teatro e contos. **Design de Jogos:** Criação de arcos de personagem profundos e ramificados para NPCs (personagens não-jogáveis) em videogames. **Narrativas Visuais:** Geração de sequências de prompts para ferramentas de IA de imagem (como Midjourney ou DALL-E) para ilustrar a evolução visual de um personagem ao longo do tempo. **Marketing e Branding:** Criação de histórias de clientes (customer success stories) ou narrativas de marca que demonstrem transformação e superação, gerando maior conexão emocional com o público. **Design Instrucional:** Aplicação da estrutura de arco de história para criar jornadas de aprendizado mais envolventes e memoráveis (Story-Smart Prompts).

## Pitfalls
**Transformação Instantânea:** O LLM pula do "antes" para o "depois" sem mostrar o processo de mudança. **Solução:** Sempre solicite explicitamente por falhas, retrocessos e o "meio bagunçado" da jornada. **Mudança Não Motivada:** A transformação do personagem não tem uma causa-e-efeito clara. **Solução:** Encadeie os prompts de forma lógica, garantindo que cada passo seja uma resposta direta ao evento anterior. **Voz Inconsistente:** O personagem se torna irreconhecível à medida que muda. **Solução:** Ancore a voz e a personalidade com traços específicos que persistem, mesmo após a transformação. **História de Fundo Esquecida:** O histórico do personagem não influencia a transformação. **Solução:** Referencie eventos passados nos prompts de transformação para que o LLM use a história de fundo como base para as lições aprendidas.

## URL
[https://consistentcharacter.ai/blog/build-ai-character-arcs-through-strategic-prompts](https://consistentcharacter.ai/blog/build-ai-character-arcs-through-strategic-prompts)
