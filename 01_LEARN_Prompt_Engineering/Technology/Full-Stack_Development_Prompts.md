# Full-Stack Development Prompts

## Description
**Prompts de Desenvolvimento Full-Stack** são técnicas de engenharia de prompt focadas em alavancar Modelos de Linguagem Grande (LLMs) para auxiliar em todas as etapas do ciclo de vida do desenvolvimento de software, abrangendo tanto o frontend quanto o backend, além de infraestrutura e testes. O objetivo é transformar a IA em um co-piloto de desenvolvimento que pode gerar código, arquitetar soluções, criar testes, configurar ambientes de implantação (CI/CD, Docker) e depurar problemas, resultando em um aumento significativo na produtividade e na qualidade do código. A eficácia desses prompts reside na capacidade de fornecer contexto detalhado, especificações técnicas claras e restrições de segurança. Eles são essenciais para automatizar tarefas repetitivas e complexas, permitindo que o desenvolvedor se concentre em lógica de negócios de alto nível.

## Examples
```
**1. Geração de Estrutura de Projeto:**
"Crie a estrutura de pastas completa para uma aplicação full-stack moderna. Frontend em Next.js (TypeScript) e Backend em FastAPI (Python). Inclua diretórios para componentes, serviços de API, modelos de banco de dados (PostgreSQL), testes unitários e configuração de ambiente. Apresente a saída em formato de árvore de diretórios Markdown."

**2. Criação de Componente Frontend:**
"Gere um componente React (TypeScript) para um formulário de login. O formulário deve ter validação de email e senha, estado de carregamento e exibição de erro. Utilize Tailwind CSS para o estilo e inclua um `handleSubmit` que simule uma chamada de API. O código deve ser modular e incluir comentários."

**3. Implementação de Endpoint Backend:**
"Desenvolva um endpoint de API em Node.js (Express) para criação de usuário. O endpoint deve receber nome, email e senha. A senha deve ser hasheada com bcrypt. Use Mongoose para interagir com um banco de dados MongoDB. Inclua validação de entrada e tratamento de erros para email duplicado. Forneça o código completo do controlador e do modelo."

**4. Configuração de Infraestrutura (Docker):**
"Crie um arquivo `docker-compose.yml` para um ambiente de desenvolvimento full-stack. Os serviços devem incluir: um frontend React, um backend Flask (Python) e um banco de dados PostgreSQL. Configure volumes persistentes para o banco de dados e mapeie as portas necessárias. Adicione um serviço de cache Redis."

**5. Geração de Testes Unitários:**
"Escreva testes unitários abrangentes usando Jest e React Testing Library para o componente de 'Carrinho de Compras'. Os testes devem cobrir: renderização inicial, adição e remoção de itens, cálculo total e o estado de carrinho vazio. Mocke as chamadas de API necessárias para buscar dados do produto."

**6. Refatoração e Otimização de Código:**
"Analise o seguinte trecho de código JavaScript e refatore-o para usar programação assíncrona com `async/await` e otimize o loop para melhor performance. Explique as mudanças e o ganho de eficiência: [INSERIR CÓDIGO AQUI]"

**7. Documentação Técnica:**
"Gere a documentação técnica para o endpoint `/api/v1/orders` do backend. A documentação deve incluir: método HTTP, URL, parâmetros de requisição (com tipos e exemplos), estrutura de resposta de sucesso (200) e códigos de erro (400, 401, 500). Use o formato OpenAPI/Swagger."

**8. Debugging e Correção de Erros:**
"O seguinte erro está ocorrendo no meu código Python/Django: `[INSERIR STACK TRACE AQUI]`. Analise o stack trace, identifique a causa raiz e forneça o trecho de código corrigido, explicando o porquê da correção."
```

## Best Practices
**1. Seja Específico e Contextual:** Sempre inclua a stack tecnológica (React, Node.js, Python, etc.), o propósito do código e o contexto do projeto (ex: "aplicação de e-commerce", "microserviço de autenticação"). **2. Defina o Formato de Saída:** Peça explicitamente o formato desejado (ex: "código em TypeScript", "estrutura de pastas em Markdown", "testes em Jest"). **3. Peça Explicações e Comentários:** Solicite que o código seja comentado e que a IA explique o raciocínio por trás das decisões de design ou segurança. **4. Itere e Refine:** Use o output inicial como base e peça refinamentos, como "Otimize este código para performance" ou "Adicione tratamento de erros para a API". **5. Inclua Restrições de Segurança:** Especifique requisitos de segurança (ex: "Use bcrypt para hashing de senhas", "Implemente proteção CSRF").

## Use Cases
nan

## Pitfalls
**1. Confiança Excessiva (Alucinações):** A IA pode gerar código que parece correto, mas contém erros lógicos ou de sintaxe sutil. **Sempre** verifique e teste o código gerado. **2. Falta de Contexto:** Prompts vagos levam a código genérico e inútil. A falta de especificação da stack, versão ou arquitetura resulta em retrabalho. **3. Ignorar Segurança:** A IA pode gerar código com vulnerabilidades de segurança (ex: injeção SQL, XSS) se não for explicitamente instruída a seguir as melhores práticas de segurança. **4. Dependência de Boilerplate:** Usar a IA apenas para código repetitivo sem entender os princípios subjacentes impede o aprendizado e a evolução do desenvolvedor. **5. Prompts Longos Demais:** Embora o contexto seja crucial, prompts excessivamente longos e complexos podem confundir a IA, levando a respostas incompletas ou fora do escopo. Mantenha o foco em uma tarefa por prompt.

## URL
[https://www.linkedin.com/pulse/ultimate-guide-ai-prompting-full-stack-development-2024-2025-patil-9n4zf](https://www.linkedin.com/pulse/ultimate-guide-ai-prompting-full-stack-development-2024-2025-patil-9n4zf)
