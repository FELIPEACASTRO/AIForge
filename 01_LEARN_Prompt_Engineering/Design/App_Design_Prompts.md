# App Design Prompts

## Description
**Prompts de Design de Aplicativos** são instruções estruturadas e detalhadas fornecidas a modelos de Inteligência Artificial (IA) generativa (como LLMs e modelos de imagem) para auxiliar ou automatizar tarefas no processo de design de interfaces de usuário (UI) e experiência do usuário (UX) para aplicativos móveis e web. Eles transformam ideias conceituais em ativos de design tangíveis, como wireframes, fluxos de usuário, especificações de estilo, código de componentes e cópia de UX. A eficácia desses prompts reside na sua capacidade de incorporar os princípios de design, as diretrizes de plataforma e as necessidades específicas do usuário, agindo como um "engenheiro de design" virtual. O objetivo é acelerar a fase de ideação, prototipagem e especificação, permitindo que os designers se concentrem em problemas de UX mais complexos e estratégicos.

## Examples
```
**1. Geração de Wireframe:**
"Crie um wireframe de baixa fidelidade para a tela de 'Checkout' de um aplicativo de comércio eletrônico. O objetivo é minimizar o atrito. Inclua os seguintes elementos: lista de itens, campo de endereço de entrega, opções de pagamento (cartão, Pix), e um botão de 'Finalizar Compra' proeminente. Use o tom 'direto e seguro'. Formato de saída: Descrição em Markdown e lista de componentes."

**2. Fluxo de Usuário (User Flow):**
"Mapeie o fluxo de usuário completo para o 'Primeiro Login e Integração' (Onboarding) de um aplicativo de meditação. O fluxo deve ter 4 etapas: 1. Tela de Boas-Vindas, 2. Seleção de Objetivos (Ex: Reduzir Estresse), 3. Permissão de Notificações, 4. Tela Inicial. Para cada etapa, sugira a cópia de UX em um tom 'calmo e encorajador'. Formato de saída: Tabela em Markdown."

**3. Especificação de Estilo e Acessibilidade:**
"Sugira uma paleta de cores e tipografia para um aplicativo de gerenciamento financeiro pessoal. A paleta deve ser baseada em tons de azul e verde, transmitindo confiança e crescimento. Garanta que todos os pares de cores de texto e fundo atendam aos padrões de acessibilidade WCAG AA. Tipografia: Uma fonte sans-serif moderna e legível. Formato de saída: Paleta de 5 cores (HEX, RGB) e 2 fontes (Nome, Peso)."

**4. Geração de Código de Componente:**
"Gere o código React Native para um componente de 'Cartão de Notificação' para um aplicativo de notícias. O cartão deve incluir: um ícone de categoria, um título (máx. 50 caracteres), um resumo (máx. 100 caracteres) e um carimbo de data/hora. O design deve seguir as diretrizes do Material Design. Formato de saída: Bloco de código React Native."

**5. Cópia de UX para Erro:**
"Escreva a cópia de UX para uma mensagem de erro que aparece quando um usuário tenta enviar um formulário sem preencher um campo obrigatório. O tom deve ser 'útil e amigável', evitando culpa. A mensagem deve indicar claramente o problema e a solução. Formato de saída: Texto da mensagem de erro e texto do botão de ação."

**6. Análise de Concorrência:**
"Analise a tela inicial dos aplicativos 'Duolingo' e 'Babbel'. Identifique os 3 principais elementos de UI que promovem o engajamento e a retenção. Sugira como podemos adaptar esses elementos para um novo aplicativo de aprendizado de idiomas focado em conversação. Formato de saída: Análise comparativa em parágrafos."
```

## Best Practices
**Estrutura do Prompt (5 C's):** Um prompt eficaz deve conter **Clareza** (o que fazer), **Contexto** (para quem e onde), **Especificidade** (detalhes técnicos e visuais), **Tom** (a voz da marca/app) e **Formato** (o tipo de saída desejada, como wireframe, código, ou texto).
**Iteração e Refinamento:** Comece com prompts simples e adicione camadas de complexidade. Use a saída do primeiro prompt como contexto para o próximo.
**Definição de Restrições:** Inclua restrições de acessibilidade (WCAG), diretrizes de plataforma (iOS Human Interface Guidelines, Material Design) e paletas de cores específicas.
**Foco no Problema de UX:** Em vez de pedir apenas um design bonito, peça à IA para resolver um problema de experiência do usuário, como "reduzir o abandono de carrinho" ou "simplificar o processo de integração".
**Uso de Dados Estruturados:** Para saídas como tabelas de recursos ou fluxos de usuário, solicite o formato de saída como JSON ou Markdown para facilitar a integração com outras ferramentas.

## Use Cases
**Ideação Rápida e Brainstorming:** Gerar rapidamente múltiplas variações de layout, paletas de cores ou conceitos de recursos para a fase inicial do projeto.
**Criação de Wireframes e Protótipos:** Transformar especificações de requisitos em esboços visuais de baixa ou média fidelidade.
**Geração de Cópia de UX:** Criar textos para botões, mensagens de erro, notificações e fluxos de integração (onboarding) que se alinhem ao tom de voz da marca.
**Especificação de Design System:** Definir e documentar componentes de UI, regras de espaçamento, tipografia e acessibilidade para um Design System.
**Tradução de Design para Código:** Gerar código de componentes de UI (ex: React, Vue, Swift) a partir de descrições de design, acelerando o handoff para o desenvolvimento.
**Análise e Otimização de UX:** Solicitar à IA que analise um fluxo de usuário existente e sugira melhorias com base em princípios de usabilidade.

## Pitfalls
**Prompts Vagos ou Genéricos:** Pedir "um design de aplicativo bonito" sem especificar o público, o objetivo ou o estilo resulta em saídas irrelevantes ou clichês.
**Ignorar o Contexto de UX:** Focar apenas na estética (UI) e negligenciar o fluxo de usuário, a hierarquia da informação e a resolução de problemas (UX) leva a designs visualmente agradáveis, mas disfuncionais.
**Excesso de Confiança na Primeira Saída:** A IA é uma ferramenta de ideação. A primeira saída raramente é a solução final. É crucial iterar, refinar e aplicar o julgamento humano de design.
**Violação de Direitos Autorais/Plágio:** Usar prompts que imitam diretamente o estilo de um aplicativo existente pode levar a problemas de propriedade intelectual. Sempre busque inspiração em princípios, não em cópias diretas.
**Falta de Especificações Técnicas:** Não incluir o framework (React Native, Flutter, Web) ou as diretrizes de design (Material Design, iOS HIG) pode gerar componentes de código ou layouts que não são implementáveis.

## URL
[https://medium.com/@uxraspberry/prompt-engineering-for-designers-a-practical-guide-what-i-learned-so-far-140d70879c7e](https://medium.com/@uxraspberry/prompt-engineering-for-designers-a-practical-guide-what-i-learned-so-far-140d70879c7e)
