# System Architecture Prompts

## Description
"System Architecture Prompts" represents a shift in focus from a simple *prompting* exercise to a **system design** discipline when building robust applications based on Large Language Models (LLMs) [1]. Rather than being an isolated prompt technique, it is an approach that integrates the LLM as a central component in a broader software architecture, where the prompt is the control point for guiding the model's reasoning. The main challenge is to build a system that can steer the probabilistic and, at times, unpredictable nature of LLMs toward reliable and accurate results [1].

Architecture prompts are often used in conjunction with patterns such as **Retrieval-Augmented Generation (RAG)**, where the prompt is *augmented* with factual context retrieved from an external knowledge base, and in **Agents** architectures, where the LLM uses the prompt to decide which tool or API to call to perform a task [1]. The goal is to transform the LLM from a fallible "know-it-all" into a "reasoning engine" that operates over a set of provided facts [1].

## Examples
```
1. **Scalability Planning (RAG-Augmented):** "Based on these system metrics [insert traffic and latency metrics], identify possible bottlenecks in the database and the microservices layer. Propose three *sharding* solutions and compare them based on cost and implementation complexity. Use the provided context to justify your choice."
2. **Software Architecture Design:** "Act as a Senior Solutions Architect. Design a *serverless* microservices architecture for an e-commerce order processing system that expects a peak of 10,000 orders per minute. The design must include: core services, *message queues*, database (choose between NoSQL or SQL and justify), and a high-level diagram in Mermaid format. Consider resilience and observability."
3. **Security Analysis:** "Analyze the following Python code snippet [insert code snippet] for SQL injection and XSS vulnerabilities. Additionally, suggest an end-to-end encryption mechanism for communication between the authentication service and the payment service, detailing the protocol (TLS 1.3, AES-256)."
4. **Design Pattern Recommendation:** "We are developing a real-time notification system that needs to send messages to millions of users via email, SMS, and *push notification*. Describe the most suitable design pattern (e.g., *Observer*, *Pub/Sub*, *Event Sourcing*) to manage message distribution and justify why it is superior to the others for this use case."
5. **Data Model Design Assistance:** "Create an optimized data schema for a relational database (PostgreSQL) for an inventory management system. The main entities are: Product, Warehouse, Supplier, and Stock Transaction. Include the primary keys, foreign keys, and the indexes necessary to optimize low-latency queries on the stock level in a specific warehouse."
6. **Comparative Technology Analysis:** "Provide a comparative analysis between Kubernetes (K8s) and AWS ECS for container orchestration, considering the following criteria: operational cost, team learning curve, ease of integration with CI/CD, and horizontal scalability capacity. Conclude with a recommendation for a startup focused on development speed."
7. **Technical Documentation Generation:** "Generate a detailed *outline* for the technical documentation of a new *Machine Learning Inference* service. The *outline* must include sections for: Architecture Overview, Data Flow Diagram, Infrastructure Requirements (CPU/GPU), *Rollback* Strategy, and Monitoring Plan (latency and error metrics)."
```

## Best Practices
- **Be Specific and Contextual:** Provide as much detail as possible about the system, metrics, parameters, and constraints. The prompt should be a reflection of the requirements documentation.
- **Use Placeholders:** Use brackets (`[]`) to clearly indicate where the LLM should insert specific information (code, metrics, descriptions), facilitating automation.
- **Enable Interaction:** Add an instruction such as "Ask questions if you need more information" to allow the LLM to request the missing context, improving the quality of the response.
- **Integrate with RAG:** To ensure accuracy and up-to-date knowledge, use the prompt in the **Contextual Prompting** stage of a RAG pipeline, providing the LLM with the factual context retrieved from your proprietary knowledge base [1].
- **Decompose the Problem:** For complex tasks, use orchestration techniques such as **Chain-of-Thought (CoT)** to force the LLM to break the problem down into logical steps before providing the final solution.

## Use Cases
- **Architecture Design:** Generation of architecture proposals (microservices, monolith, *serverless*), including high-level diagrams and technology-choice justifications.
- **Vulnerability Analysis:** Review of code and system design to identify security flaws and suggest protection mechanisms (encryption, authentication).
- **Performance and Scalability Optimization:** Identification of bottlenecks, suggestion of *sharding* strategies, load balancing, and database query optimization.
- **Data Modeling:** Creation of optimized data schemas for different types of databases (SQL, NoSQL) based on query requirements and data volume.
- **Documentation Generation:** Creation of *outlines*, templates, and drafts of technical documentation, disaster recovery plans (DRP), and business continuity plans (BCP).
- **Technology Evaluation:** Comparative analysis of technology *stacks* (e.g., *frameworks*, cloud providers, programming languages) based on defined criteria (cost, performance, support).
- **Compliance Assurance:** Verification that the architecture design meets industry-specific regulations (e.g., GDPR, LGPD, HIPAA).

## Pitfalls
- **Lack of Detail:** Vague prompts lead to generic and useless responses. Software architecture is complex and requires clear specifications (metrics, technology *stack*, cost/time constraints).
- **Blind Trust (*Over-reliance*):** Assuming that the LLM output is infallible, especially on critical matters such as security, cost, and regulatory compliance. The LLM output should be treated as a suggestion from a consultant and always validated by a human architect.
- **Ignoring Proprietary Context:** Not providing the system-specific context (internal data, legacy APIs) in the prompt. The LLM has no knowledge of the internal details of your organization.
- **Absence of RAG/Grounding:** Trying to obtain factual or domain-specific answers without augmenting the prompt with retrieved data, resulting in **hallucinations** or outdated information about your system.
- **Monolithic Prompts:** Trying to solve a complex architecture problem in a single prompt. Orchestration and decomposition of the problem into smaller, chained prompts are essential.

## URL
[https://medium.com/@vi.ha.engr/the-architects-guide-to-llm-system-design-from-prompt-to-production-8be21ebac8bc](https://medium.com/@vi.ha.engr/the-architects-guide-to-llm-system-design-from-prompt-to-production-8be21ebac8bc)
