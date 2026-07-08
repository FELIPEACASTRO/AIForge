# Role-Based and Pattern-Driven System Architecture Prompts

## Description

Prompt Engineering for System Architecture Design is an emerging discipline that uses Large Language Models (LLMs) to assist software architects across a variety of tasks, from generating initial designs from requirements to analyzing and reviewing existing architectures. The technique is based on clearly defining a **role** for the LLM (e.g., "Senior Cloud Architect") and applying **prompt patterns** (e.g., Chain-of-Thought, Role-Based Templates) to guide the model through a structured design process. This allows LLMs to act as "coaches" or assistants, increasing productivity and the quality of architectural decisions.

## Statistics

- **Adoption:** The use of LLMs in software architecture tasks is growing strongly, with a sharp increase in academic publications in 2024 and 2025 (Source: *Software Architecture Meets LLMs: A Systematic Literature Review*, 2025).
- **Automation:** 71% of academic works use LLMs in an automated or semi-automated way for architecture tasks.
- **Effectiveness:** Studies show that LLMs frequently **outperform baselines** in tasks such as classifying design decisions and retrieving traceability links.
- **Common Models:** GPT-4, GPT-3.5, and BERT are the most widely used models in research in this area.

## Features

- **Design Generation:** Creation of architecture designs from functional and non-functional requirements.
- **Classification and Detection:** Identification of design patterns, architectural tactics, and design decisions in code or documentation.
- **Code/Architecture Review:** Analysis of diagrams and code based on defined policies and standards (e.g., `Code review with policy`).
- **Decision Assistance:** Help with selecting, evaluating, and capturing architectural decisions (ADRs - Architecture Decision Records).
- **Transformation:** Conversion of sketches (whiteboard/sketch) into formal digital diagrams (requires multimodal LLMs).

## Use Cases

- **Architecture Generation from Requirements:** Create an initial microservices design from a list of user stories.
- **Technical Debt Analysis:** Prioritize and explain issues identified in formal code analyses for non-technical audiences.
- **Architecture Coaching:** Continuous dialogue to apply patterns such as "branching by abstraction" in serverless functions.
- **Test Scaffolding:** Generation of critical integration tests from API schemas (e.g., OpenAPI/Swagger).

## Integration

### Best Practices:
1. **Role Definition:** Begin the prompt by defining the LLM's role (e.g., "Act as a Senior Solutions Architect with 15 years of experience in distributed systems and AWS").
2. **Context and Constraints:** Provide as much context as possible, including non-functional requirements (scalability, security, cost) and technological constraints (languages, cloud providers).
3. **Output Structure:** Request the output in a structured format (e.g., Markdown, PlantUML, JSON) and define the expected sections (e.g., Component Diagram, Justifications, Risks).
4. **Iteration and Refinement (Chain-of-Thought):** Use follow-up prompts to refine the design, requesting justifications, risk analysis, or alternatives (e.g., "Now, analyze the security risks of the authentication component and propose 3 mitigations.").

### Prompt Example (Role-Based Template):

```
**Role:** You are a Senior Software Architect specializing in event-driven and serverless architectures.

**Task:** Design the high-level architecture for a new Order Processing Service.

**Requirements:**
1. **Functional:** Receive orders, validate inventory, notify the billing system.
2. **Non-Functional:** High availability (99.99%), scalability to 1000 orders/second, cost optimized (serverless preferred).
3. **Constraints:** Must use AWS, Python for processing functions, and a NoSQL database for the order catalog.

**Output Instructions:**
1. **Component Diagram:** Describe the main components (e.g., API Gateway, Lambda, SQS, DynamoDB) and their interactions.
2. **Technology Justification:** Explain why the serverless/event-driven architecture is the best choice.
3. **Risk Analysis:** Identify the main scalability bottleneck and the mitigation strategy.
4. **Format:** Use Markdown for the description and a list format for the components.
```

## URL

https://github.com/mikaelvesavuori/chatgpt-architecture-coach
