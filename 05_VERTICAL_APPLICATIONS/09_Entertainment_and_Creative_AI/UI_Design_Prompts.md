# UI Design Prompts

## Description
Prompt engineering para Design de Interface do Usuário (UI) e Experiência do Usuário (UX) é a arte de criar comandos específicos e detalhados para modelos de Inteligência Artificial (IA) generativa (como LLMs e modelos de imagem) com o objetivo de auxiliar em todas as fases do processo de design. Isso inclui desde a pesquisa inicial e a criação de personas até a geração de wireframes, protótipos, cópia de UX e análise de fluxo. A técnica visa maximizar a utilidade e a precisão das saídas da IA, transformando-a em uma ferramenta de co-criação e otimização para designers. A eficácia reside em incorporar os princípios de design (como clareza, contexto e restrições) diretamente no prompt.

## Examples
```
**1. Geração de Componente Específico:**
"Crie um componente de cartão de produto para um e-commerce de moda sustentável. O cartão deve incluir uma imagem principal, o nome do produto (fonte sans-serif, 16px), o preço em negrito, um selo de 'Eco-Friendly' e um botão de 'Adicionar ao Carrinho' com ícone de folha. O estilo deve ser minimalista e usar uma paleta de cores terrosas."

**2. Análise de Fluxo e Otimização (Requer input de wireframe/descrição):**
"Analise o seguinte fluxo de cadastro de usuário para um aplicativo de investimento. O fluxo tem 5 etapas. Sugira melhorias para reduzir o atrito e a taxa de abandono, focando na simplificação dos campos de entrada e na clareza da proposta de valor em cada etapa."

**3. Criação de Persona Detalhada:**
"Crie 3 personas detalhadas para um aplicativo de entrega de comida gourmet. Considere fatores como idade, localização, hábitos de uso de tecnologia, principais frustrações com aplicativos atuais e objetivos ao usar o novo serviço. Foque em como aumentar a retenção de usuários."

**4. Geração de Cópia de UX (UX Writing):**
"Escreva a cópia de UX para uma mensagem de erro que aparece quando o usuário tenta enviar um formulário sem preencher um campo obrigatório. A cópia deve ser amigável, útil e indicar claramente qual campo precisa de atenção. Use um tom de voz casual e encorajador."

**5. Brainstorming de Funcionalidades:**
"Liste 10 funcionalidades inovadoras para um aplicativo de planejamento de viagens focado em 'viagens de última hora'. Para cada funcionalidade, descreva brevemente o problema que ela resolve e o elemento de UI necessário para implementá-la."

**6. Recomendação de Design System:**
"Recomende uma paleta de cores e um par de fontes (uma para títulos, outra para corpo de texto) apropriados para um aplicativo B2B de gerenciamento de projetos. O design deve transmitir profissionalismo, confiança e eficiência. Justifique suas escolhas com base na psicologia das cores e na legibilidade."

**7. Explicação de Conceitos para Stakeholders:**
"Explique o conceito de 'Arquitetura da Informação' para um executivo não-técnico. Use o exemplo de um supermercado para ilustrar como a organização do conteúdo afeta a experiência do usuário e as vendas."
```

## Best Practices
**1. Clareza e Especificidade:** Seja o mais detalhado possível. Em vez de "Crie um botão", diga "Crie um botão primário de 'Comprar Agora' com cantos arredondados, cor de fundo azul (#007BFF), texto branco e um ícone de carrinho de compras à esquerda."
**2. Forneça Contexto:** Inclua o máximo de informação sobre o projeto, público-alvo e a fase do design. Defina a persona da marca e o objetivo da tela.
**3. Use Restrições (Constraints):** Defina limites claros. Especifique o sistema de design (ex: Material Design, iOS Human Interface Guidelines), a paleta de cores, ou o número de elementos.
**4. Refinamento Iterativo:** Use a saída da IA como ponto de partida. Peça refinamentos como: "Agora, torne este design mais acessível para usuários com baixa visão, aumentando o contraste e o tamanho da fonte."
**5. Adote uma Persona:** Peça à IA para agir como um especialista: "Aja como um Designer de UX Sênior da Google e avalie este fluxo de checkout."

## Use Cases
**1. Ideação e Brainstorming:** Geração rápida de conceitos de tela, fluxos de usuário alternativos e funcionalidades inovadoras no início de um projeto.
**2. Otimização de Fluxos e Usabilidade:** Análise de wireframes ou descrições de fluxo para identificar pontos de atrito e sugerir melhorias de usabilidade e acessibilidade.
**3. Criação de Conteúdo de UX (UX Writing):** Geração de microcópias, mensagens de erro, textos de onboarding e chamadas para ação (CTAs) que se alinham ao tom de voz da marca.
**4. Prototipagem Rápida:** Criação de componentes de UI e layouts básicos que servem como ponto de partida para protótipos de baixa ou média fidelidade.
**5. Pesquisa e Análise:** Geração de perguntas para entrevistas com usuários, criação de roteiros de testes de usabilidade e síntese de dados de pesquisa em personas e *user journeys*.
**6. Design System e Estilo:** Sugestão de paletas de cores, tipografia e diretrizes de estilo que se encaixam na identidade visual e nos requisitos de acessibilidade do projeto.

## Pitfalls
**1. Prompts Vagos ou Genéricos:** Solicitações como "Crie uma tela de login bonita" resultam em saídas genéricas e inutilizáveis. A falta de especificidade é o erro mais comum.
**2. Ignorar o Contexto do Usuário:** Não fornecer informações sobre o público-alvo, a plataforma (iOS, Android, Web) ou o objetivo do produto leva a designs desalinhados com as necessidades reais.
**3. Dependência Excessiva da Primeira Saída:** Tratar a IA como um designer final em vez de um assistente. O resultado da IA é um rascunho que *sempre* requer revisão, iteração e validação humana.
**4. Falha em Definir Restrições Técnicas:** Não especificar o framework (ex: React, Vue) ou a biblioteca de componentes (ex: Tailwind, Bootstrap) pode gerar código ou sugestões de design que são difíceis de implementar.
**5. Não Usar Iteração:** Enviar um prompt complexo de uma só vez em vez de dividir a tarefa em etapas menores e refinadas (ex: 1. Crie o layout. 2. Ajuste as cores. 3. Escreva a cópia).

## URL
[https://www.uxpin.com/studio/blog/prompt-engineering-for-designers/](https://www.uxpin.com/studio/blog/prompt-engineering-for-designers/)
