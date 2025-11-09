# World-Building Prompts (Prompts de Construção de Mundo)

## Description
**Prompts de Construção de Mundo (World-Building Prompts)** são uma técnica de Engenharia de Prompt focada em instruir Modelos de Linguagem Grande (LLMs) a criar ambientes, cenários e universos ficcionais detalhados e coerentes. Em vez de pedir uma única resposta, o método envolve uma abordagem **hierárquica e iterativa**, onde o prompt inicial estabelece as regras e o contexto fundamental do mundo, e prompts subsequentes refinam os detalhes em camadas (geografia, história, cultura, personagens, etc.). O objetivo é aproveitar a capacidade do LLM de manter a **consistência contextual** em longas sessões, garantindo que os detalhes gerados em níveis inferiores (como a personalidade de um personagem ou o material de construção de um edifício) estejam alinhados com o contexto global estabelecido no início. Esta técnica é fundamental para criadores de conteúdo, escritores, desenvolvedores de jogos e qualquer pessoa que precise de um ambiente ficcional rico e pronto para uso.

## Examples
```
1. **Criação de Contexto Global (Passo 1):**
   \`\`\`
   Crie o contexto fundamental para um mundo de fantasia.
   **Gênero:** Fantasia Sombria Pós-Apocalíptica.
   **Regra Mágica:** A magia é alimentada pela emoção, mas cada uso drena a vitalidade do usuário, manifestando-se como uma doença física.
   **Tecnologia:** Nível de tecnologia da Revolução Industrial, mas com máquinas movidas a vapor e aprimoradas por cristais mágicos.
   **Conflito Central:** A luta entre as últimas cidades-estado fortificadas e as hordas de "Vazios" (seres sem emoção) que vagam pelas terras devastadas.
   \`\`\`

2. **Refinamento de Localização (Passo 2):**
   \`\`\`
   Com base no contexto global, descreva a cidade-estado de **"Aethelgard"**.
   **Localização:** Construída dentro de um antigo vulcão extinto, usando basalto e metal reciclado.
   **Governo:** Uma teocracia militar liderada por um "Conselho de Sacerdotes de Ferro".
   **Cultura:** Uma sociedade obcecada por ordem e supressão emocional para evitar o uso de magia e a atração dos Vazios.
   **Estrutura:** Descreva 3 bairros distintos e seus propósitos.
   \`\`\`

3. **Criação de Personagem com Contexto (Passo 3):**
   \`\`\`
   Crie um personagem que vive no bairro **"O Fosso"** de Aethelgard (o bairro mais pobre e perigoso).
   **Nome:** Kael.
   **Função:** Um "Engenheiro de Vapor" que secretamente usa magia emocional para consertar máquinas.
   **Personalidade:** Cínico, desconfiado, mas com um forte senso de justiça.
   **Falha:** Sua magia o está deixando cego lentamente.
   **Prompt:** Descreva Kael, sua aparência, sua motivação e um breve encontro com ele.
   \`\`\`

4. **Geração de Item/Artefato:**
   \`\`\`
   Crie um artefato importante para o mundo de Aethelgard.
   **Nome:** O "Coração de Basalto".
   **Função:** Um cristal mágico que absorve a emoção de uma área, tornando-a segura, mas deixando as pessoas apáticas.
   **História:** Descreva sua origem e como ele é usado pelo Conselho de Sacerdotes de Ferro.
   \`\`\`

5. **Criação de Regra Social/Lei:**
   \`\`\`
   Qual é a lei mais importante e a punição mais severa em Aethelgard?
   **Lei:** A "Lei da Quietude Emocional".
   **Punição:** Ser exilado para as terras devastadas, onde os Vazios o encontrarão.
   **Prompt:** Escreva um pequeno aviso público sobre esta lei, como seria visto em um mural da cidade.
   \`\`\`

6. **Prompt de Diálogo (Teste de Consistência):**
   \`\`\`
   Escreva um diálogo de 5 turnos entre Kael (o Engenheiro Cínico) e um Sacerdote de Ferro sobre uma máquina quebrada. O diálogo deve refletir a desconfiança de Kael e a obsessão do Sacerdote por regras.
   \`\`\`

7. **Prompt de Geografia:**
   \`\`\`
   Descreva a paisagem imediatamente fora dos muros de Aethelgard. Inclua detalhes sobre a flora, fauna e as ruínas da civilização anterior.
   \`\`\`

8. **Prompt de Fato Histórico:**
   \`\`\`
   Crie um evento histórico crucial que levou à formação de Aethelgard. O evento deve envolver um surto de magia emocional descontrolada.
   \`\`\`

9. **Prompt de Cultura/Religião:**
   \`\`\`
   Descreva um ritual ou festival anual em Aethelgard. O ritual deve ser uma celebração da ordem e da supressão emocional.
   \`\`\`

10. **Prompt de Estrutura (JSON):**
    \`\`\`
    Gere uma lista JSON de 5 edifícios típicos no bairro **"O Fosso"** de Aethelgard. Para cada edifício, inclua: "nome", "função", "material_principal" e "tamanho_m2".
    \`\`\`
```

## Best Practices
**Consistência Hierárquica:** Comece com o contexto global (tipo de mundo, regras básicas) e refine em camadas (regiões, cidades, bairros, edifícios, personagens). A consistência do nível superior deve ser reforçada nos níveis inferiores. **Contexto Rico em Linguagem Natural:** Use descrições ricas e não estruturadas para os elementos de nível superior (como a história do mundo ou o caráter da cidade), pois os LLMs são excelentes em manter o contexto e o tom. **Saída Estruturada para Detalhes:** Para elementos que precisam ser usados em um formato de jogo ou banco de dados (tamanhos, funções, inventário), solicite explicitamente a saída estruturada (JSON, listas). **Injeção de Caráter:** Use o contexto do mundo e do local para influenciar a criação de personagens, evitando a "positividade implacável" dos LLMs. Peça por falhas, limitações e personalidades culturalmente apropriadas. **Iteração e Ajuste:** Use o LLM para gerar listas de ideias (tipos de cidades, nomes) e, em seguida, use suas próprias escolhas para refinar e preencher os detalhes. O processo é colaborativo.

## Use Cases
**Escrita Criativa e Narrativa:** Criação de cenários ricos e detalhados para romances, contos, roteiros e histórias em quadrinhos. **Desenvolvimento de Jogos (RPG e Videogames):** Geração rápida de lore, história, facções, cidades e personagens para jogos de RPG de mesa (como D&D) ou para o design inicial de mundos de videogames. **Design de Experiência (UX/UI):** Criação de cenários e personas ficcionais para testar a usabilidade de produtos em contextos simulados. **Treinamento de Modelos de IA:** Geração de grandes volumes de dados textuais consistentes para treinar ou refinar modelos de IA em tarefas de narrativa e coerência contextual. **Educação:** Uso em aulas de escrita criativa ou história para simular a criação de civilizações e culturas.

## Pitfalls
**Inconsistência Contextual:** O LLM pode "esquecer" regras ou detalhes estabelecidos em prompts anteriores, especialmente em sessões longas. É crucial reintroduzir o contexto principal (ex: "Lembre-se da Regra Mágica:...") em prompts subsequentes. **"Positividade Implacável":** A tendência do LLM de criar personagens e cenários excessivamente positivos ou genéricos. É necessário solicitar ativamente falhas, conflitos, sujeira e realismo. **Geração Desestruturada:** Pedir detalhes complexos sem especificar o formato pode resultar em texto difícil de analisar ou usar em um sistema estruturado (como um jogo). **Sobrecarga de Informação:** Tentar definir muitos detalhes de uma vez no prompt inicial pode diluir o foco do LLM. A abordagem deve ser hierárquica e gradual. **Falta de Conflito:** Um mundo sem conflitos inerentes (sociais, políticos, ambientais) será chato. O prompt inicial deve estabelecer um conflito central.

## URL
[https://ianbicking.org/blog/2023/02/world-building-with-gpt](https://ianbicking.org/blog/2023/02/world-building-with-gpt)
