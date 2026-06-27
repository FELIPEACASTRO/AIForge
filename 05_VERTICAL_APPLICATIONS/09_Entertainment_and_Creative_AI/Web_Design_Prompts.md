# Web Design Prompts

## Description
**Prompts de Design Web** são instruções estruturadas e detalhadas fornecidas a modelos de Inteligência Artificial (IA) generativa (como LLMs ou modelos de imagem) para auxiliar ou automatizar tarefas no processo de design e desenvolvimento web. Essa técnica de *Prompt Engineering* no contexto de design web abrange desde a geração de ideias conceituais, wireframes e layouts visuais (UI/UX) até a criação de código front-end, conteúdo otimizado para SEO e elementos de branding. O objetivo principal é acelerar o ciclo de design, permitindo que designers e desenvolvedores iterem rapidamente em conceitos e se concentrem em aspectos mais complexos e estratégicos do projeto. A eficácia reside na capacidade de traduzir requisitos de negócios e princípios de design em comandos claros e acionáveis para a IA.

## Examples
```
**1. Geração de Wireframe (Estrutura):**
\`\`\`
Crie um wireframe em HTML/CSS para a página inicial de um SaaS de gestão de projetos. O público-alvo são gerentes de equipe. O layout deve ser limpo, focado em um CTA proeminente "Teste Grátis" no topo e incluir seções para: 1) Prova Social (Logos de Clientes), 2) Lista de Recursos Principais (com ícones), e 3) Tabela de Preços (3 planos). O design deve ser responsivo e mobile-first.
\`\`\`

**2. Design Visual (Geração de Imagem/Conceito):**
\`\`\`
Design de página de destino (landing page) para um produto de café especial, estilo visual minimalista e orgânico, com paleta de cores terrosas (marrom, bege, verde escuro). Foco em fotografia de alta qualidade do produto e tipografia serifada elegante. O layout deve ter uma dobra superior (hero section) com um grande título e um botão "Compre Agora". Renderização fotorrealista, 16:9, --ar 16:9
\`\`\`

**3. Otimização de Conteúdo e SEO:**
\`\`\`
Atue como um especialista em SEO e UX Writer. Gere 5 opções de meta description (máx. 160 caracteres) e 5 opções de título H1 para uma página de produto que vende "Tênis de Corrida Sustentáveis". O foco deve ser em sustentabilidade, conforto e o termo-chave "tênis de corrida sustentáveis".
\`\`\`

**4. Refinamento de Componente (CSS):**
\`\`\`
Gere o código CSS para um botão de Call-to-Action (CTA) que chame a atenção. O botão deve ter um gradiente animado sutil de azul (#007bff) para ciano (#00bcd4), bordas arredondadas (12px), e uma sombra suave que se expande ao passar o mouse (hover effect).
\`\`\`

**5. Criação de Persona e Jornada do Usuário:**
\`\`\`
Crie uma persona detalhada para um site de notícias financeiras. Inclua nome, idade, ocupação, objetivos (goals), frustrações (pain points) e um breve cenário de uso (user story) de como essa persona interagiria com o site para encontrar informações sobre investimentos em criptomoedas.
\`\`\`

**6. Geração de Estrutura de Navegação:**
\`\`\`
Sugira a estrutura de navegação (sitemap) de nível 1 e 2 para um site de uma universidade. O site deve atender a três públicos principais: 1) Futuros Alunos, 2) Alunos Atuais e 3) Professores/Funcionários. A navegação deve ser clara e intuitiva.
\`\`\`
```

## Best Practices
**1. Seja Específico e Contextualizado:** Sempre inclua o **objetivo** do design (vender, informar, capturar leads), o **público-alvo** e o **tom de voz** desejado.
**2. Defina o Formato de Saída:** Especifique se você deseja código (HTML/CSS/JS), um esboço de layout (wireframe), um design visual (imagem) ou apenas a estrutura de conteúdo.
**3. Use Restrições e Estilos:** Inclua restrições de design (paleta de cores, tipografia, acessibilidade) e referências de estilo (minimalista, futurista, retrô, flat design).
**4. Itere e Refine:** Comece com um prompt amplo e use prompts de refinamento (por exemplo, "Mude a cor principal para #1A73E8" ou "Aumente o espaço em branco na seção de depoimentos").
**5. Integre SEO e UX:** Peça explicitamente para otimizar elementos para SEO (meta tags, alt text, estrutura de cabeçalhos) e para a experiência do usuário (CTAs claros, navegação intuitiva, responsividade).

## Use Cases
**1. Prototipagem Rápida (Wireframing):** Gerar rapidamente a estrutura de layout (HTML/CSS) para testar diferentes fluxos de usuário e hierarquias de conteúdo.
**2. Geração de Conceitos Visuais (Moodboards):** Criar imagens conceituais de alta fidelidade para apresentar a paleta de cores, estilo e atmosfera de um projeto antes de iniciar o desenvolvimento.
**3. Otimização de Conteúdo Web:** Gerar títulos, meta descriptions, alt texts e microcópias (textos de botões, mensagens de erro) otimizados para SEO e conversão.
**4. Criação de Componentes de UI:** Gerar código para componentes específicos (formulários, carrosséis, menus de navegação) com base em requisitos de design e frameworks específicos (ex: Bootstrap, Tailwind CSS).
**5. Definição de Estratégia de UX:** Criar personas de usuário, jornadas do cliente e mapas de site (sitemaps) para alinhar o design com os objetivos de negócio e as necessidades do usuário.

## Pitfalls
**1. Prompts Vagos ou Genéricos:** Pedir apenas "Crie um site legal" resulta em saídas irrelevantes. A falta de contexto (público, objetivo, estilo) é o erro mais comum.
**2. Confiar Demais no Primeiro Resultado:** A IA é uma ferramenta de rascunho. O output inicial raramente é o produto final. A falha em iterar e refinar o prompt leva a designs medíocres.
**3. Ignorar a Acessibilidade:** Não incluir requisitos de acessibilidade (WCAG) no prompt pode gerar designs visualmente agradáveis, mas inutilizáveis para pessoas com deficiência.
**4. Misturar Requisitos de Forma e Função:** Tentar gerar o código e o conceito visual em um único prompt complexo pode sobrecarregar o modelo. É melhor separar a geração de conceito visual (imagem) da geração de código (texto).
**5. Falha em Especificar a Tecnologia:** Não indicar a tecnologia de saída (ex: "Gere em React e Tailwind CSS" vs. "Gere em HTML e CSS puro") pode resultar em código inútil.

## URL
[https://www.websitebuilderexpert.com/building-websites/ai-prompts/](https://www.websitebuilderexpert.com/building-websites/ai-prompts/)
