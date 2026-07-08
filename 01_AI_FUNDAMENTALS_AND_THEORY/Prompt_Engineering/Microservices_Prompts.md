# Microservices Prompts

## Description
"Microservices Prompts" refers to the application of Prompt Engineering to optimize and accelerate the microservices development lifecycle. The technique leverages the ability of Large Language Models (LLMs) to act as "domain experts" (e.g., Software Architect, Senior Developer) to generate high-quality code, configurations, tests, documentation, and analyses specific to the microservices environment (e.g., Spring Boot, Kafka, Kubernetes). The focus is on providing highly structured and contextual prompts that include functional and non-functional requirements (such as security, performance, and resilience) to ensure that the generated code is robust and adheres to distributed architecture best practices.

## Examples
```
**1. Complete Boilerplate Generation:**
```
Act as a Senior Spring Boot Developer. Generate an exhaustive boilerplate for a RESTful API Spring Boot application, ready for enterprise deployment. Include: 1. An `/api/v1/products` endpoint for CRUD operations on the `Product` entity (id: Long, name: String, description: String, price: BigDecimal, stock: Integer). 2. Architecture: controller, service, repository, model, config, and util packages. 3. Versions: Spring Boot 3.2.x, Java 21, Maven 3.9.x. 4. Database: PostgreSQL with Spring Data JPA. 5. Documentation: Swagger/OpenAPI integration. 6. Best Practices: SOLID adherence, use of DTOs with validation, and layered architecture.
```

**2. SQL Query Optimization:**
```
Analyze the following PostgreSQL query and suggest optimizations. Assume the `orders` and `customers` tables are large. `SELECT * FROM orders WHERE customer_id IN (SELECT id FROM customers WHERE city = 'New York');` Explain the performance problems and provide an optimized query using `JOIN`. Discuss the role of the `EXPLAIN` command in identifying bottlenecks.
```

**3. Security Implementation (JWT/RBAC):**
```
Generate a Spring Security configuration for a Spring Boot REST API that uses JWT-based authentication and role-based access control (RBAC). Define the `ADMIN` and `USER` roles. Protect endpoints such as `/api/admin/**` for `ADMIN` and `/api/user/**` for `USER`. Include a basic JWT filter and the `UserDetailsService` implementation.
```

**4. Unit Test Generation (Mockito/JUnit):**
```
Generate a JUnit 5 unit test class for a Spring Boot `UserService` with a `registerUser(User user)` method that saves a user and `findByUsername(String username)` that retrieves a user. The `UserService` depends on a `UserRepository` interface. Use `@ExtendWith(MockitoExtension.class)` and `@BeforeEach` to configure a `mock UserRepository`. Write a test method that verifies that save is called on the mocked repository with the correct user object.
```

**5. Code Refactoring and Analysis:**
```
Act as a Senior Software Architect. Review the provided Spring Boot service class, responsible for user management. [Insert the Java service class code here]. Identify any 'code smells' (e.g., long method, duplicated code), performance bottlenecks, or areas for structural improvement (e.g., SOLID adherence). Suggest concrete refactoring strategies and explain your reasoning step by step.
```

**6. Resilience and Inter-Service Communication:**
```
Act as a Microservices Architect. For a Spring Boot microservice that makes synchronous REST calls to another internal service (e.g., an `OrderService` calling a `PaymentService`), suggest patterns to improve resilience and performance. Focus on the Circuit Breaker pattern (e.g., using Resilience4j) and client-side load balancing (e.g., using Spring Cloud LoadBalancer). Provide a conceptual Java code snippet.
```
```

## Best Practices
**Role-Playing Definition:** Begin the prompt with "Act as a [Domain Expert]" (e.g., Architect, Senior Developer, Debugger) to steer the tone and knowledge of the LLM. **Context and Version Specification:** Include the specific technology, framework, and versions (e.g., Spring Boot 3.2.x, Java 21, PostgreSQL) to ensure the relevance of the generated code and configurations. **Detailed Output Structure:** Use numbered lists or bullets to detail the output requirements (e.g., Architecture, Logging, Documentation) to ensure the LLM covers all aspects. **Focus on Non-Functionals:** Include non-functional requirements (e.g., SOLID, performance, security, resilience) to elevate the quality of the generated code beyond basic functionality. **Tool Integration:** Mention specific tools and libraries (e.g., HikariCP, Resilience4j, JUnit 5, Mockito) to obtain ready-to-use integration code.

## Use Cases
**Rapid Boilerplate Generation:** Create the initial structure of a new microservice in minutes. **Performance Optimization:** Analyze and optimize database queries, JVM configurations, and caching strategies. **Test Generation:** Create complex unit and integration tests, including mocks and specific configurations. **Security and Validation:** Generate security configurations (JWT, RBAC) and DTOs with robust input validation. **Code Refactoring and Analysis:** Identify "code smells" and suggest structural improvements in existing code. **Automated Documentation:** Generate Javadoc or OpenAPI/Swagger annotations for APIs.

## Pitfalls
**Lack of Context:** Overly generic prompts lead to code that does not fit the company's architecture or standards. **Ignoring Non-Functional Requirements:** Focusing only on functionality can result in code with security, performance, or maintainability problems. **Over-Reliance:** Blindly trusting the generated code without human review, which can introduce subtle bugs or vulnerabilities. **Prompt Injection:** Risk of vulnerability in microservices that use LLMs to generate user-facing content (e.g., product descriptions), requiring input validation and guardrails. **Prompt Maintenance:** Complex prompts become a code asset that needs version control and refinement, just like source code.

## URL
[https://medium.com/@prashantjadhav/strategic-ai-prompt-engineering-for-spring-boot-microservices-46bcae26bc79](https://medium.com/@prashantjadhav/strategic-ai-prompt-engineering-for-spring-boot-microservices-46bcae26bc79)
