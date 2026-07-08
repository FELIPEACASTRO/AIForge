# API Design Prompts

## Description
**API Design Prompts** is a Prompt Engineering technique focused on using Large Language Models (LLMs) to assist or automate the process of designing Application Programming Interfaces (APIs). The main goal is to translate natural-language business requirements (such as user stories or product requirements documents) into structured, ready-to-use API specifications, such as OpenAPI (Swagger) documents or JSON Schema.

This technique leverages the ability of LLMs to understand intent, extract entities, define data models, and generate the endpoint structure (routes, HTTP methods, parameters, and responses) based on textual descriptions. It is a key component of the **AI-First API Design** approach, where the API specification is generated before the code, accelerating prototyping and ensuring design consistency.

Its effectiveness lies in the prompt's ability to provide sufficient context (API purpose, target audience, main entities) and to require a structured, verifiable output, allowing the AI to act as an assistant software architect.

## Examples
```
**1. OpenAPI Specification Generation (Swagger):**
```
Act as a senior API Architect.
**Task:** Generate a complete OpenAPI 3.0 specification for a task management API.
**Requirements:**
- Main entity: 'Task' (id, title, description, due_date, status [pending, completed], user_id).
- Endpoints:
  - GET /tasks: List all tasks with support for filtering by 'status' and pagination.
  - POST /tasks: Create a new task.
  - GET /tasks/{id}: Get details of a specific task.
  - PUT /tasks/{id}: Update an existing task.
  - DELETE /tasks/{id}: Delete a task.
- Authentication: Use Bearer Token (OAuth2).
**Output Format:** YAML.
```

**2. Data Model Definition (JSON Schema):**
```
**Task:** Create the JSON Schema for the 'Product' data model of an e-commerce system.
**Attributes:**
- name (string, required, minLength: 3)
- sku (string, required, format: alphanumeric with hyphens)
- price (number, required, format: float, minimum: 0.01)
- stock_quantity (integer, required, minimum: 0)
- categories (array of strings, optional)
- is_available (boolean, required)
**Output Format:** JSON Schema Draft 2020-12.
```

**3. Design Refinement and Error Handling:**
```
**Context:** I have the following OpenAPI specification (paste the YAML/JSON).
**Task:** Review the error response section for the POST /users endpoint.
**Requirement:** Add a 409 Conflict status code for the case where a user tries to register with an already existing email. The response body must include an 'error_code' field and a 'message' in English.
**Output Format:** The revised 'paths' section for the /users endpoint.
```

**4. Endpoint Generation from a User Story:**
```
**User Story:** As a user, I want to be able to reset my password by providing my email and receiving a reset link by email.
**Task:** Design the RESTful endpoint (method, route, request body, and response) needed to implement this user story.
**Output Format:** Markdown description with the request and response model in JSON.
```

**5. Documentation and Code Examples:**
```
**Context:** The endpoint is GET /orders/{orderId} and returns the 'Order' object.
**Task:** Generate a Python code example (using the 'requests' library) that makes a call to this endpoint, including Bearer Token authentication, and prints the delivery status.
**Output Format:** Complete Python code block.
```
```

## Best Practices
**1. Clarity and Specificity:** Clearly define the API's purpose, the business domain, and the functional requirements. Use precise language and avoid ambiguity.
**2. Structure and Format:** Request the output in a structured format (e.g., OpenAPI/Swagger, JSON Schema) to facilitate integration and validation.
**3. Business Context:** Provide the business context and validation rules so the AI can design data models and endpoints that reflect the reality of the application.
**4. Iteration and Refinement:** Use the AI's initial output as a draft. Request specific refinements, such as adding pagination, authentication, or error handling.
**5. Style Compliance:** Include references to API style guides (if any) to ensure consistency of naming and design patterns.

## Use Cases
**1. Rapid API Prototyping:** Quickly generate OpenAPI specifications from user stories to create *mock servers* and allow *frontend* development to begin in parallel.
**2. Documentation Generation:** Automatically create detailed and consistent API documentation (e.g., parameter descriptions, response examples) from a specification draft.
**3. Legacy System Modernization:** Analyze the documentation or code of legacy APIs to generate a modern OpenAPI specification, facilitating migration and integration.
**4. Design Validation:** Use the AI to review an existing API specification, identifying inconsistencies, security flaws, or style guide violations.
**5. *Boilerplate* Code Generation:** Create code templates (e.g., data model classes, endpoint controllers) in specific languages (Python, Java, Node.js) directly from the generated specification.

## Pitfalls
**1. Ambiguity in Requirements:** Vague or contradictory requirements lead to incorrect or incomplete API specifications. The AI cannot guess the business intent.
**2. Over-reliance:** Treating the AI's output as final without human review. API design requires nuances of security, performance, and business context that the AI may overlook.
**3. Lack of Style Context:** Not providing a style guide or design standards results in inconsistent APIs (e.g., mixed use of `camelCase` and `snake_case`).
**4. Ignoring Security:** The AI may generate functional specifications but fail to implement robust security mechanisms or forget crucial authorization details.
**5. Domain Complexity:** For highly complex or niche-specific business logic, the AI may struggle to model entities and relationships correctly, requiring extensive refinement prompts.

## URL
[https://kinde.com/learn/ai-for-software-engineering/using-ai-for-apis/ai-first-api-design-generating-openapi-specs-from-natural-language-requirements/](https://kinde.com/learn/ai-for-software-engineering/using-ai-for-apis/ai-first-api-design-generating-openapi-specs-from-natural-language-requirements/)
