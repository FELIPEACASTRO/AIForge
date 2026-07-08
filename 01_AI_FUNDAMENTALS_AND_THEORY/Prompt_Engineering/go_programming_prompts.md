# Go Programming Prompts

## Description
**Go Programming Prompts** (or Golang) refer to prompt engineering instructions optimized for interacting with Large Language Models (LLMs) with the goal of generating, analyzing, debugging, or refactoring code in the Go language. This technique focuses on leveraging AI for Go-specific development tasks, such as creating concurrent microservices, applying idiomatic design patterns, and solving complex concurrency problems, such as *deadlocks* and *goroutine leaks*.

The effectiveness of these prompts lies in including Go-specific context, such as the need for **idiomatic** code (following *Effective Go*), explicit error handling, and the correct use of concurrency features (goroutines and channels). By providing clear guidelines on the architecture, libraries (e.g., Gin, gRPC), and Go conventions, the developer maximizes the accuracy and usefulness of the AI-generated code.

## Examples
```
1.  **RESTful Microservice Generation:**
    > "You are a senior software engineer with extensive experience in Go. Develop a step-by-step tutorial, including complete and explanatory code examples, that demonstrates how to build a RESTful microservice in Go to manage users, using the **Gin** framework and a **PostgreSQL** database connection. The code must follow the idiomatic Go style, with explicit error handling and the recommended project structure."

2.  **Concurrency Debugging:**
    > "You are a Go expert. Analyze the following Go code snippet (insert code) that is exhibiting an intermittent *deadlock*. Create an interactive debugging guide for intermediate-level developers, explaining the root cause of the problem and providing the corrected solution. The guide should include the steps to use the **race detector** and **pprof** to diagnose the problem."

3.  **Design Pattern Application:**
    > "Create a detailed guide for Go developers, exploring the **Factory** design pattern for creating different types of database connections (MySQL, MongoDB). Include Go code examples that illustrate the application of the pattern idiomatically, focusing on the interface and dependency injection."

4.  **Integration Testing Strategy:**
    > "Develop a detailed step-by-step tutorial for beginners on how to implement effective integration tests for a Go service that interacts with an external payment service. The tutorial should use **test containers** to simulate the external service and ensure high coverage and reliability, with code examples for setting up and running the tests."

5.  **Performance Optimization:**
    > "You are a Go performance optimization expert. Analyze the following code (insert code) and provide a practical tutorial on how to optimize memory usage and avoid unnecessary allocations, focusing on the efficient use of *slices* and *maps*. Provide the optimized version of the code and explain the improvements."

6.  **API Documentation Generation:**
    > "Create a comprehensive, practical guide for developing API documentation in Go projects. The guide should focus on automatically generating documentation from code comments, using the **Swag** tool. Include examples of correctly formatted code comments and the commands needed to generate and host the documentation."

7.  **Complex System Simulation:**
    > "Create a Go module that simulates the management of an e-commerce. The module should include structs for `Product` (ID, Name, Price, QuantityInStock) and `Order` (ID, Slice of Product IDs, Status, CreationDate). The prompt should request the implementation of a `ProcessOrder` function that uses **goroutines** and **channels** to simulate the asynchronous processing of orders and the updating of stock."
```

## Best Practices
*   **Be Idiomatic:** Always instruct the AI to follow the idiomatic Go style, referencing *Effective Go* and the naming conventions (`gofmt`).
*   **Specify Concurrency:** When dealing with concurrent tasks, be explicit about the use of **goroutines** and **channels**, and request the inclusion of synchronization mechanisms (e.g., `sync.Mutex`, `sync.WaitGroup`).
*   **Explicit Error Handling:** Require that the generated code use Go's explicit error handling, returning errors instead of using exceptions or *panics* (except in cases of unrecoverable errors).
*   **Define the Context and Dependencies:** Specify the libraries, frameworks (e.g., Gin, gRPC), and the Go version to be used.
*   **Request Tests:** Ask the AI to generate unit and integration tests for the code, following Go's standard `testing` package.

## Use Cases
*   **Rapid Prototype Development:** Generating *scaffolding* for microservices, APIs, and command-line (CLI) tools.
*   **Solving Concurrency Problems:** Diagnosing and fixing *deadlocks*, *race conditions*, and *goroutine leaks*.
*   **Refactoring and Optimization:** Suggestions for refactoring non-idiomatic code or optimizing memory and CPU usage.
*   **Learning and Tutorials:** Creating code examples for design patterns, data structures, and advanced Go features.
*   **Documentation Generation:** Creating API documentation and usage guides from the source code.

## Pitfalls
*   **Non-Idiomatic Code:** The AI may generate code that works but does not follow Go conventions (e.g., excessive use of classes/inheritance instead of composition, non-idiomatic error handling).
*   **Incorrect Concurrency:** Concurrency is complex; the AI may introduce subtle *race conditions* or *deadlocks* if the prompt is not rigorous enough about synchronization.
*   **Over-Reliance:** Blindly trusting the generated code without critical review, especially regarding security and performance.
*   **Lack of Context:** The AI may not have the full context of the project, resulting in code that does not integrate well with the existing architecture.
*   **Library Hallucinations:** The AI may reference libraries or functions that are obsolete or no longer exist in the current version of Go.

## URL
[https://www.cabare.dev.br/topics/go](https://www.cabare.dev.br/topics/go)
