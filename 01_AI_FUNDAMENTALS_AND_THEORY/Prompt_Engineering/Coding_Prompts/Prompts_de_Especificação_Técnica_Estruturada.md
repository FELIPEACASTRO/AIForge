# Prompts de Especificação Técnica Estruturada

## Description

A técnica de Prompts de Especificação Técnica Estruturada é uma abordagem de Engenharia de Prompt que visa gerar documentos técnicos detalhados e precisos (como Especificações de Requisitos de Software - SRS, ou Documentos de Arquitetura de Sistema) usando Modelos de Linguagem Grande (LLMs). Ela se baseia na aplicação de princípios de engenharia de prompt, como a atribuição de papéis (ex: "Você é um Engenheiro de Software Sênior"), a definição de contexto, a especificação de restrições e o uso de formatos de saída estruturados (ex: Markdown, JSON, ou um template de documento específico). O objetivo é transformar entradas de alto nível (ex: uma ideia de produto) em documentação técnica acionável e de alta qualidade, garantindo que o LLM atue como um especialista e siga rigorosamente as diretrizes de engenharia.

## Statistics

Embora não haja estatísticas padronizadas de LLM disponíveis publicamente para esta técnica, a eficácia é amplamente suportada por estudos de caso e artigos de engenharia de prompt (Infomineo, 2025). A aplicação de técnicas de estruturação (como Role-Assignment e Constraint Specification) demonstrou aumentar a precisão e a relevância da saída em até 40% em tarefas complexas de raciocínio e documentação, em comparação com prompts genéricos (Infomineo, 2025). A adoção de templates de prompt por plataformas de produtividade (ClickUp, 2025) e ferramentas de documentação (WriteDoc.ai) indica uma alta taxa de uso e aceitação na indústria de desenvolvimento de software.

## Features

- **Atribuição de Papel de Especialista:** O LLM é instruído a agir como um Engenheiro de Software, Arquiteto de Sistemas ou Gerente de Produto para garantir o tom e a profundidade técnica apropriados.
- **Estrutura de Saída Definida:** O prompt exige um formato de saída específico (ex: seções numeradas, tabelas, formato de documento padrão) para garantir a consistência e a facilidade de uso.
- **Inclusão de Restrições:** Permite a inclusão de requisitos de compatibilidade (hardware/software), persona do usuário final e objetivos de desempenho.
- **Geração de Documentação Completa:** Capaz de gerar esboços, seções específicas ou documentos técnicos completos, como SRS, Especificações de Design e Documentos de Arquitetura.

## Use Cases

- **Geração de SRS (Software Requirements Specification):** Criação de documentos formais de requisitos para novos projetos de software.
- **Documentação de Arquitetura de Sistema:** Auxílio na descrição de componentes, interações e decisões de design de sistemas complexos.
- **Especificações de Design de API:** Definição de endpoints, payloads e comportamento de APIs REST ou GraphQL.
- **Criação de Casos de Teste:** Geração de cenários de teste detalhados a partir de requisitos funcionais.
- **Documentação de Design System:** Criação de especificações técnicas para componentes de UI/UX para desenvolvedores e designers.

## Integration

**Melhores Práticas:**
1.  **Defina o Papel:** Comece com `Você é um [Engenheiro de Software Sênior/Arquiteto de Sistemas]...`
2.  **Forneça Contexto:** Descreva o produto, o público-alvo e o objetivo principal.
3.  **Especifique a Estrutura:** Use listas numeradas ou um template de documento (ex: "Inclua as seções: Introdução, Requisitos Funcionais, Requisitos Não Funcionais, Design de Alto Nível").
4.  **Adicione Restrições:** Inclua detalhes técnicos cruciais (ex: `Compatível com Windows 10`, `Tempo de resposta inferior a 500ms`).

**Exemplo de Prompt (Template ClickUp Adaptado):**

`Você é um Engenheiro de Software Sênior. Preciso desenvolver especificações técnicas detalhadas para um [tipo de produto: aplicativo móvel de saúde e bem-estar] que será usado por [persona: profissionais de 25 anos interessados em saúde]. O documento deve ser abrangente e facilmente compreendido por desenvolvedores e designers. O objetivo principal é [propósito: rastrear o sono e a ingestão de água].`

`O documento de especificação técnica deve incluir as seguintes seções detalhadas:`
`1. Introdução (Visão Geral do Produto e Público-Alvo)`
`2. Requisitos Funcionais (Ex: Login/Logout, Rastreamento de Sono, Registro de Água)`
`3. Requisitos Não Funcionais (Ex: Desempenho, Segurança, Usabilidade)`
`4. Design de Alto Nível (Componentes e Interações)`
`5. Requisitos de Compatibilidade (Ex: iOS e Android)`

`Use um tom profissional e técnico. Certifique-se de que os requisitos sejam SMART (Específicos, Mensuráveis, Alcançáveis, Relevantes, com Prazo).`

## URL

https://infomineo.com/artificial-intelligence/prompt-engineering-techniques-examples-best-practices-guide/