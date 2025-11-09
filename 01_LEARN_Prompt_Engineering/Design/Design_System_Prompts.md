# Design System Prompts

## Description
"Design System Prompts" (Prompts de Sistema de Design) é uma técnica de engenharia de prompt que utiliza modelos de linguagem grande (LLMs) e outras IAs generativas para criar, manter, documentar ou aplicar um **Design System (DS)**. O conceito central é fornecer o Design System (incluindo tokens de design, componentes e diretrizes de marca) como **contexto** ou **guarda-corpo** para a IA. Isso garante que a saída da IA (seja código, design ou documentação) seja inerentemente consistente com os padrões de design e marca estabelecidos. A evolução recente (2024-2025) foca na integração de Design Systems como "conhecimento" estruturado em ferramentas de construção de aplicativos e editores de código com IA, muitas vezes utilizando protocolos como o Model Context Protocol (MCP) para passar o contexto de forma legível por máquina.

## Examples
```
1. **Geração de Componente (Alto Nível):**
`"Crie um componente de 'Card de Notificação' para o Figma. Ele deve usar o token de cor 'color-brand-primary' para o cabeçalho, o token de tipografia 'font-body-medium' para o corpo, e o componente 'Button-Primary' para a ação. A estrutura deve ser: Ícone (à esquerda), Título, Corpo do Texto, e Botão de Ação (à direita)."`

2. **Geração de Layout (Com DS como Contexto):**
`"Usando apenas os componentes disponíveis no Design System [Nome do DS], gere o código React para uma página de 'Configurações de Perfil'. O layout deve incluir um componente 'Avatar', três campos de 'Input-Text' para Nome, Email e Senha, e um 'Button-Primary' para salvar. Aplique o espaçamento 'spacing-large' entre os elementos."`

3. **Criação de DS (Fidelidade Progressiva):**
`"Estou criando um aplicativo de gerenciamento de tarefas. Crie um Design System de baixa fidelidade, perguntando-me sobre a paleta de cores principal, a família de fontes e os três componentes de UI mais importantes. Pergunte-me uma pergunta de cada vez."`

4. **Auditoria de Consistência:**
`"Analise o seguinte trecho de código HTML/CSS. Identifique todas as instâncias onde as cores ou tamanhos de fonte não correspondem aos tokens definidos no nosso Design System (tokens: color-text-default: #333, font-size-body: 16px). Sugira a correção usando os tokens."`

5. **Documentação de Componente:**
`"Gere a documentação de uso (em Markdown) para o componente 'Modal de Confirmação'. Inclua a descrição, exemplos de uso (com código de exemplo), e uma lista de todas as 'props' (título, mensagem, onConfirm, onCancel) com seus tipos e valores padrão."`

6. **Geração de Tokens de Design:**
`"Crie um arquivo JSON de tokens de design para uma marca com foco em sustentabilidade. Defina tokens para cores primárias (verde escuro, verde claro), secundárias (bege, branco), tipografia (uma fonte serifada para títulos, uma sans-serif para corpo) e espaçamento (small, medium, large)."`

7. **Refatoração de Código:**
`"Refatore o código CSS abaixo para substituir todos os valores hexadecimais por tokens de design correspondentes do nosso Design System. Se um token não existir, use o token mais próximo e sinalize a alteração. Código a ser refatorado: \`background-color: #007bff; padding: 20px;\`"`

8. **Verificação de Acessibilidade (Accessibility Check):**
`"Analise o componente 'Button-Secondary' do nosso Design System. Verifique se a combinação de cor de fundo ('color-background-secondary') e cor do texto ('color-text-on-secondary') atende ao contraste mínimo WCAG AA (4.5:1). Se não atender, sugira o token de cor de texto mais próximo que atenda."`
```

## Best Practices
**Fornecer o DS como Contexto:** Em vez de apenas descrever o que você quer, forneça o Design System completo (tokens, componentes, diretrizes) como parte do contexto de conhecimento da ferramenta de IA. **Uso de Fidelidade Progressiva (Low-to-High Fidelity):** Comece com prompts para gerar um DS de baixa fidelidade (estrutura básica, cores primárias, tipografia). Revise e refine, e então use prompts para evoluir para média e alta fidelidade, adicionando detalhes e complexidade. **Estrutura Clara do Prompt:** O prompt deve ser claro sobre o **objetivo**, o **contexto do DS** (se não for fornecido automaticamente), e o **formato de saída** desejado (código, descrição, componente Figma, etc.). **Utilizar o MCP (Model Context Protocol):** Para ferramentas de código/design que suportam o MCP (como o Figma Dev Mode MCP Server), use essa integração para passar o contexto do DS de forma estruturada e legível por máquina, em vez de depender apenas de texto. **Foco em "Guardrails":** Use o DS para atuar como "guardrails" (corrimãos) para a IA, limitando as escolhas criativas a elementos pré-aprovados, garantindo consistência e qualidade.

## Use Cases
**Geração de Componentes Consistentes:** Gerar novos componentes de UI que aderem automaticamente aos tokens de design e convenções de nomenclatura do DS. **Criação de Telas/Layouts:** Gerar layouts de tela inteiros ou protótipos interativos usando apenas componentes existentes do DS. **Documentação Automatizada:** Gerar documentação técnica e de uso para novos componentes do DS. **Refatoração e Migração:** Usar a IA para refatorar código legado para que utilize componentes e tokens do DS. **Auditoria de Consistência:** Pedir à IA para auditar um design ou código existente e identificar violações das diretrizes do DS.

## Pitfalls
**Confiança Excessiva na "Verdade" da IA:** Assumir que a saída da IA é 100% precisa e consistente com o DS sem revisão humana. **Prompts Ambíguos:** Usar linguagem vaga que permite à IA fazer suposições que violam as diretrizes do DS. **Contexto Insuficiente:** Não fornecer contexto suficiente sobre o DS, resultando em saídas genéricas ou inconsistentes. **Foco Apenas em Texto:** Tentar descrever um DS complexo apenas com texto em vez de usar integrações estruturadas (como MCP) ou arquivos de configuração (como tokens JSON).

## URL
[https://www.youngleaders.tech/p/how-to-prompt-create-a-design-system-](https://www.youngleaders.tech/p/how-to-prompt-create-a-design-system-)
