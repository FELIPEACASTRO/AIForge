# Geração de Prompts de Teste (Test Generation Prompts)

## Description
A técnica de **Geração de Prompts de Teste** (Test Generation Prompts) consiste em elaborar instruções estruturadas e detalhadas para que um Grande Modelo de Linguagem (LLM) gere artefatos de teste de software. Isso inclui casos de teste, scripts de automação, planos de teste, dados de teste e até mesmo relatórios de bugs. O objetivo principal é alavancar a capacidade do LLM de entender requisitos complexos e transformá-los em artefatos de teste acionáveis, aumentando a cobertura, a eficiência e a velocidade do ciclo de vida de QA (Quality Assurance). Esta técnica é fundamental na engenharia de software moderna, especialmente no contexto de metodologias ágeis e DevOps, onde a velocidade e a qualidade são cruciais. O uso eficaz desses prompts requer a inclusão de contexto, formato de saída desejado e a especificação clara do tipo de teste e dos critérios de aceitação.

## Examples
```
**1. Geração de Casos de Teste Funcionais (Tabela Markdown):**
\`\`\`
Aja como um Engenheiro de QA Sênior. Gere 10 casos de teste funcionais para o recurso de "Login de Usuário" com base no seguinte requisito: "O usuário deve ser capaz de fazer login com um email válido e uma senha de 8 a 16 caracteres. Tentativas de login falhas devem exibir uma mensagem de erro genérica. Após 3 tentativas falhas, a conta deve ser bloqueada por 5 minutos."
Formato de Saída: Tabela Markdown com colunas: ID, Título do Teste, Pré-condições, Passos, Resultado Esperado, Tipo (Positivo/Negativo).
\`\`\`

**2. Geração de Script de Automação (Python/Selenium):**
\`\`\`
Escreva um script de teste de automação em Python usando a biblioteca Selenium para verificar a funcionalidade de "Adicionar Item ao Carrinho" em um site de e-commerce. O script deve: 1. Navegar até a URL do produto. 2. Clicar no botão "Adicionar ao Carrinho". 3. Verificar se o número de itens no ícone do carrinho é atualizado para 1.
URL do Produto: [URL do Produto]
\`\`\`

**3. Geração de Dados de Teste (JSON):**
\`\`\`
Gere 5 conjuntos de dados de teste no formato JSON para testar a API de registro de novos usuários. Inclua 2 casos de sucesso (dados válidos) e 3 casos de falha (e.g., email inválido, senha muito curta, campo obrigatório ausente).
Estrutura JSON esperada: {"nome": "string", "email": "string", "senha": "string"}.
\`\`\`

**4. Geração de Testes de Segurança (OWASP Top 10):**
\`\`\`
Com base no seguinte trecho de código (ou descrição da funcionalidade): [Trecho de Código/Descrição], identifique e gere 3 cenários de teste de segurança que abordem as vulnerabilidades do OWASP Top 10 (e.g., Injeção SQL, XSS). Para cada cenário, forneça o vetor de ataque e o resultado esperado.
\`\`\`

**5. Geração de Testes de Performance (JMeter Plan):**
\`\`\`
Crie um plano de teste de carga para a funcionalidade de "Busca de Produtos". O teste deve simular 500 usuários simultâneos por 10 minutos. O tempo de resposta médio esperado é inferior a 500ms. Forneça os passos para configurar este teste no Apache JMeter, incluindo o Thread Group e o HTTP Request Sampler.
\`\`\`

**6. Geração de Cenários de Teste de Usabilidade (Heurísticas de Nielsen):**
\`\`\`
Analise a seguinte interface de usuário (descreva a interface ou forneça um link) e gere 5 cenários de teste de usabilidade com base nas Heurísticas de Nielsen (e.g., Visibilidade do Status do Sistema, Correspondência entre o Sistema e o Mundo Real).
Interface: [Descrição da Interface]
\`\`\`
```

## Best Practices
**1. Fornecer Contexto Completo (Contextualização):** Inclua o máximo de detalhes possível sobre o sistema, o módulo, a funcionalidade e o ambiente de teste. Use a documentação do usuário, requisitos ou trechos de código como entrada. **2. Definir o Formato de Saída (Estrutura):** Especifique o formato exato que você espera (e.g., tabela Markdown, JSON, formato Gherkin, ou um script de código específico como Python/Selenium). **3. Especificar o Tipo de Teste (Intenção):** Seja explícito sobre o tipo de teste desejado (e.g., funcional, de unidade, de integração, de segurança, de performance, de usabilidade). **4. Incluir Restrições e Critérios de Aceitação:** Mencione quaisquer restrições (e.g., "apenas testes de caminho feliz", "cobrir todos os casos de erro de validação") e os critérios de aceitação para o sucesso do teste. **5. Iterar e Refinar (Refinamento Contínuo):** Use a saída do LLM como ponto de partida. Refine o prompt com base nos resultados iniciais para cobrir lacunas ou corrigir imprecisões. **6. Usar a Persona (Role-Playing):** Peça ao LLM para assumir o papel de um "Engenheiro de QA Sênior" ou "Especialista em Segurança" para obter resultados mais focados e de alta qualidade.

## Use Cases
**1. Aceleração da Criação de Casos de Teste:** Geração rápida de um grande volume de casos de teste a partir de requisitos de usuário (User Stories) ou especificações funcionais. **2. Criação de Scripts de Automação:** Geração de código inicial (e.g., Python, Java, JavaScript) para testes de unidade, integração ou UI (Interface do Usuário) usando frameworks como Selenium, Cypress ou Playwright. **3. Geração de Dados de Teste:** Criação de conjuntos de dados sintéticos, válidos e inválidos, para testar APIs e formulários, garantindo a cobertura de diferentes cenários de entrada. **4. Identificação de Lacunas de Cobertura:** Análise de um conjunto existente de testes e requisitos para sugerir cenários de teste adicionais que aumentem a cobertura e reduzam o risco. **5. Elaboração de Planos e Estratégias de Teste:** Geração de planos de teste estruturados, incluindo escopo, recursos, cronogramas e tipos de teste a serem executados. **6. Geração de Testes Específicos (Segurança e Performance):** Criação de cenários de teste focados em segurança (e.g., injeção, XSS) ou performance (e.g., testes de carga e estresse).

## Pitfalls
**1. Confiança Excessiva na Saída do LLM:** Assumir que os testes gerados são perfeitos ou completos. O LLM pode gerar testes sintaticamente corretos, mas semanticamente incorretos ou incompletos. **2. Falta de Contexto Específico:** Usar prompts vagos ou genéricos. Isso leva a casos de teste superficiais, que não cobrem as regras de negócio ou as especificidades do sistema. **3. Ignorar Casos de Borda e Negativos:** Focar apenas em "caminhos felizes" (happy paths). É crucial solicitar explicitamente testes negativos, de exceção e de casos de borda. **4. Não Especificar o Formato:** Receber a saída em um formato inconsistente ou difícil de integrar com ferramentas de QA (e.g., texto corrido em vez de JSON ou Gherkin). **5. Alucinações de Requisitos:** O LLM pode "alucinar" requisitos ou funcionalidades que não existem, gerando testes irrelevantes. Sempre valide os testes gerados contra a documentação real. **6. Não Incluir Critérios de Aceitação:** A ausência de resultados esperados claros no prompt pode levar a testes ambíguos ou não verificáveis.

## URL
[https://www.practitest.com/resource-center/blog/chatgpt-prompts-for-software-testing/](https://www.practitest.com/resource-center/blog/chatgpt-prompts-for-software-testing/)
