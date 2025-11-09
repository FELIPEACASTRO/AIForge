# API Design Prompts

## Description
**API Design Prompts** (Prompts de Design de API) é uma técnica de Engenharia de Prompt focada em utilizar Modelos de Linguagem Grande (LLMs) para auxiliar ou automatizar o processo de design de Interfaces de Programação de Aplicações (APIs). O objetivo principal é traduzir requisitos de negócio em linguagem natural (como histórias de usuário ou documentos de requisitos de produto) em especificações de API estruturadas e prontas para uso, como documentos OpenAPI (Swagger) ou JSON Schema.

Esta técnica aproveita a capacidade dos LLMs de compreender a intenção, extrair entidades, definir modelos de dados e gerar a estrutura de endpoints (rotas, métodos HTTP, parâmetros e respostas) com base em descrições textuais. É um componente chave na abordagem de **Design de API com Prioridade em IA (AI-First API Design)**, onde a especificação da API é gerada antes do código, acelerando o prototipagem e garantindo a consistência do design.

A eficácia reside na capacidade do prompt de fornecer contexto suficiente (função da API, público-alvo, entidades principais) e exigir uma saída estruturada e validável, permitindo que a IA atue como um arquiteto de software assistente.

## Examples
```
**1. Geração de Especificação OpenAPI (Swagger):**
```
Aja como um Arquiteto de API sênior.
**Tarefa:** Gere uma especificação OpenAPI 3.0 completa para uma API de gerenciamento de tarefas.
**Requisitos:**
- Entidade principal: 'Task' (id, title, description, due_date, status [pending, completed], user_id).
- Endpoints:
  - GET /tasks: Listar todas as tarefas com suporte a filtragem por 'status' e paginação.
  - POST /tasks: Criar uma nova tarefa.
  - GET /tasks/{id}: Obter detalhes de uma tarefa específica.
  - PUT /tasks/{id}: Atualizar uma tarefa existente.
  - DELETE /tasks/{id}: Excluir uma tarefa.
- Autenticação: Use Bearer Token (OAuth2).
**Formato de Saída:** YAML.
```

**2. Definição de Modelo de Dados (JSON Schema):**
```
**Tarefa:** Crie o JSON Schema para o modelo de dados 'Product' de um e-commerce.
**Atributos:**
- name (string, obrigatório, minLength: 3)
- sku (string, obrigatório, formato: alfanumérico com hífens)
- price (number, obrigatório, formato: float, mínimo: 0.01)
- stock_quantity (integer, obrigatório, mínimo: 0)
- categories (array de strings, opcional)
- is_available (boolean, obrigatório)
**Formato de Saída:** JSON Schema Draft 2020-12.
```

**3. Refinamento de Design e Tratamento de Erros:**
```
**Contexto:** Tenho a seguinte especificação OpenAPI (cole o YAML/JSON).
**Tarefa:** Revise a seção de respostas de erro para o endpoint POST /users.
**Requisito:** Adicione um código de status 409 Conflict para o caso de um usuário tentar se registrar com um email já existente. O corpo da resposta deve incluir um campo 'error_code' e uma 'message' em português.
**Formato de Saída:** A seção 'paths' revisada para o endpoint /users.
```

**4. Geração de Endpoint a Partir de História de Usuário:**
```
**História de Usuário:** Como usuário, quero poder redefinir minha senha fornecendo meu email e recebendo um link de redefinição por email.
**Tarefa:** Projete o endpoint RESTful (método, rota, corpo da requisição e resposta) necessário para implementar esta história de usuário.
**Formato de Saída:** Descrição em Markdown com o modelo de requisição e resposta em JSON.
```

**5. Documentação e Exemplos de Código:**
```
**Contexto:** O endpoint é GET /orders/{orderId} e retorna o objeto 'Order'.
**Tarefa:** Gere um exemplo de código em Python (usando a biblioteca 'requests') que faça uma chamada a este endpoint, incluindo a autenticação Bearer Token, e imprima o status da entrega.
**Formato de Saída:** Bloco de código Python completo.
```
```

## Best Practices
**1. Clareza e Especificidade:** Defina claramente o objetivo da API, o domínio de negócio e os requisitos funcionais. Use linguagem precisa e evite ambiguidades.
**2. Estrutura e Formato:** Solicite a saída em um formato estruturado (e.g., OpenAPI/Swagger, JSON Schema) para facilitar a integração e validação.
**3. Contexto de Negócio:** Forneça o contexto de negócio e as regras de validação para que a IA possa projetar modelos de dados e endpoints que reflitam a realidade da aplicação.
**4. Iteração e Refinamento:** Use a saída inicial da IA como rascunho. Peça refinamentos específicos, como a adição de paginação, autenticação ou tratamento de erros.
**5. Conformidade com Estilos:** Inclua referências a guias de estilo de API (se houver) para garantir a consistência de nomenclatura e padrões de design.

## Use Cases
**1. Prototipagem Rápida de API:** Gerar rapidamente especificações OpenAPI a partir de histórias de usuário para criar *mock servers* e permitir que o desenvolvimento *frontend* comece em paralelo.
**2. Geração de Documentação:** Criar automaticamente documentação de API detalhada e consistente (e.g., descrições de parâmetros, exemplos de resposta) a partir de um rascunho de especificação.
**3. Modernização de Sistemas Legados:** Analisar a documentação ou o código de APIs legadas para gerar uma especificação OpenAPI moderna, facilitando a migração e a integração.
**4. Validação de Design:** Usar a IA para revisar uma especificação de API existente, identificando inconsistências, falhas de segurança ou violações de guias de estilo.
**5. Geração de Código *Boilerplate*:** Criar modelos de código (e.g., classes de modelo de dados, controladores de endpoint) em linguagens específicas (Python, Java, Node.js) diretamente da especificação gerada.

## Pitfalls
**1. Ambiguidade nos Requisitos:** Requisitos vagos ou contraditórios levam a especificações de API incorretas ou incompletas. A IA não pode adivinhar a intenção de negócio.
**2. Super-revisão (Over-reliance):** Tratar a saída da IA como final sem revisão humana. O design de API requer nuances de segurança, desempenho e contexto de negócio que a IA pode negligenciar.
**3. Falta de Contexto de Estilo:** Não fornecer um guia de estilo ou padrões de design resulta em APIs inconsistentes (e.g., uso misto de `camelCase` e `snake_case`).
**4. Ignorar Segurança:** A IA pode gerar especificações funcionais, mas falhar em implementar mecanismos de segurança robustos ou esquecer detalhes cruciais de autorização.
**5. Complexidade de Domínio:** Para lógica de negócio altamente complexa ou específica de um nicho, a IA pode ter dificuldade em modelar as entidades e relações corretamente, exigindo prompts de refinamento extensivos.

## URL
[https://kinde.com/learn/ai-for-software-engineering/using-ai-for-apis/ai-first-api-design-generating-openapi-specs-from-natural-language-requirements/](https://kinde.com/learn/ai-for-software-engineering/using-ai-for-apis/ai-first-api-design-generating-openapi-specs-from-natural-language-requirements/)
