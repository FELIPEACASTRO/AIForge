# Microservices Prompts

## Description
"Microservices Prompts" refere-se à aplicação da Engenharia de Prompt para otimizar e acelerar o ciclo de vida de desenvolvimento de microsserviços. A técnica aproveita a capacidade de Large Language Models (LLMs) de atuar como "especialistas de domínio" (ex: Arquiteto de Software, Desenvolvedor Sênior) para gerar código, configurações, testes, documentação e análises de alta qualidade, específicos para o ambiente de microsserviços (ex: Spring Boot, Kafka, Kubernetes). O foco está em fornecer prompts altamente estruturados e contextuais que incluem requisitos funcionais e não-funcionais (como segurança, desempenho e resiliência) para garantir que o código gerado seja robusto e aderente às melhores práticas de arquitetura distribuída.

## Examples
```
**1. Geração de Boilerplate Completo:**
```
Atue como um Desenvolvedor Sênior Spring Boot. Gere um boilerplate exaustivo para uma aplicação RESTful API Spring Boot, pronta para implantação empresarial. Inclua: 1. Um endpoint `/api/v1/products` para operações CRUD na entidade `Product` (id: Long, name: String, description: String, price: BigDecimal, stock: Integer). 2. Arquitetura: pacotes controller, service, repository, model, config e util. 3. Versões: Spring Boot 3.2.x, Java 21, Maven 3.9.x. 4. Banco de Dados: PostgreSQL com Spring Data JPA. 5. Documentação: Integração Swagger/OpenAPI. 6. Melhores Práticas: Aderência a SOLID, uso de DTOs com validação, e arquitetura em camadas.
```

**2. Otimização de Consulta SQL:**
```
Analise a seguinte consulta PostgreSQL e sugira otimizações. Assuma que a tabela `orders` e `customers` são grandes. `SELECT * FROM orders WHERE customer_id IN (SELECT id FROM customers WHERE city = 'New York');` Explique os problemas de desempenho e forneça uma consulta otimizada usando `JOIN`. Discuta o papel do comando `EXPLAIN` para identificar gargalos.
```

**3. Implementação de Segurança (JWT/RBAC):**
```
Gere uma configuração de segurança Spring Security para uma REST API Spring Boot que usa autenticação baseada em JWT e controle de acesso baseado em função (RBAC). Defina as funções `ADMIN` e `USER`. Proteja endpoints como `/api/admin/**` para `ADMIN` e `/api/user/**` para `USER`. Inclua um filtro JWT básico e a implementação `UserDetailsService`.
```

**4. Geração de Testes de Unidade (Mockito/JUnit):**
```
Gere uma classe de teste de unidade JUnit 5 para um `UserService` Spring Boot com um método `registerUser(User user)` que salva um usuário e `findByUsername(String username)` que recupera um usuário. O `UserService` depende de uma interface `UserRepository`. Use `@ExtendWith(MockitoExtension.class)` e `@BeforeEach` para configurar um `mock UserRepository`. Escreva um método de teste que verifique se o salvamento é chamado no repositório simulado com o objeto de usuário correto.
```

**5. Refatoração e Análise de Código:**
```
Atue como um Arquiteto de Software Sênior. Revise a classe de serviço Spring Boot fornecida, responsável pelo gerenciamento de usuários. [Insira o código da classe de serviço Java aqui]. Identifique quaisquer 'code smells' (ex: método longo, código duplicado), gargalos de desempenho ou áreas para melhoria estrutural (ex: aderência a SOLID). Sugira estratégias de refatoração concretas e explique seu raciocínio passo a passo.
```

**6. Resiliência e Comunicação Inter-Serviços:**
```
Atue como um Arquiteto de Microsserviços. Para um microsserviço Spring Boot que faz chamadas REST síncronas para outro serviço interno (ex: um `OrderService` chamando um `PaymentService`), sugira padrões para melhorar a resiliência e o desempenho. Concentre-se no padrão Circuit Breaker (ex: usando Resilience4j) e balanceamento de carga do lado do cliente (ex: usando Spring Cloud LoadBalancer). Forneça um snippet de código Java conceitual.
```
```

## Best Practices
**Definição de Papel (Role-Playing):** Começar o prompt com "Atue como um [Especialista de Domínio]" (ex: Arquiteto, Desenvolvedor Sênior, Debugger) para direcionar o tom e o conhecimento do LLM. **Especificação de Contexto e Versão:** Incluir a tecnologia, framework e versões específicas (ex: Spring Boot 3.2.x, Java 21, PostgreSQL) para garantir a relevância do código e das configurações geradas. **Estrutura de Saída Detalhada:** Usar listas numeradas ou bullets para detalhar os requisitos de saída (ex: Arquitetura, Logging, Documentação) para garantir que o LLM cubra todos os aspectos. **Foco em Não-Funcionais:** Incluir requisitos não-funcionais (ex: SOLID, desempenho, segurança, resiliência) para elevar a qualidade do código gerado além da funcionalidade básica. **Integração de Ferramentas:** Mencionar ferramentas e bibliotecas específicas (ex: HikariCP, Resilience4j, JUnit 5, Mockito) para obter código de integração pronto para uso.

## Use Cases
**Geração Rápida de Boilerplate:** Criar a estrutura inicial de um novo microsserviço em minutos. **Otimização de Desempenho:** Analisar e otimizar consultas de banco de dados, configurações de JVM e estratégias de cache. **Geração de Testes:** Criar testes de unidade e integração complexos, incluindo mocks e configurações específicas. **Segurança e Validação:** Gerar configurações de segurança (JWT, RBAC) e DTOs com validação de entrada robusta. **Refatoração e Análise de Código:** Identificar "code smells" e sugerir melhorias estruturais em código existente. **Documentação Automatizada:** Gerar Javadoc ou anotações OpenAPI/Swagger para APIs.

## Pitfalls
**Falta de Contexto:** Prompts muito genéricos levam a código que não se encaixa na arquitetura ou padrões da empresa. **Ignorar Requisitos Não-Funcionais:** Focar apenas na funcionalidade pode resultar em código com problemas de segurança, desempenho ou manutenibilidade. **Dependência Excessiva:** Confiar cegamente no código gerado sem revisão humana, o que pode introduzir bugs sutis ou vulnerabilidades. **Injeção de Prompt (Prompt Injection):** Risco de vulnerabilidade em microsserviços que usam LLMs para geração de conteúdo voltado para o usuário (ex: descrições de produtos), exigindo validação de entrada e guardrails. **Manutenção de Prompts:** Prompts complexos se tornam um ativo de código que precisa de controle de versão e refinamento, assim como o código-fonte.

## URL
[https://medium.com/@prashantjadhav/strategic-ai-prompt-engineering-for-spring-boot-microservices-46bcae26bc79](https://medium.com/@prashantjadhav/strategic-ai-prompt-engineering-for-spring-boot-microservices-46bcae26bc79)
