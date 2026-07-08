# Rust Programming Prompts

## Description
Rust Programming Prompts are highly specialized prompt engineering instructions designed to leverage large language models (LLMs) to assist with software development in the Rust language. Because of Rust's unique nature, with its system of *ownership*, *borrowing*, and *lifetimes*, generic prompts are ineffective. This technique focuses on providing detailed context and requiring the AI to act as a Rust expert, addressing specific challenges such as performance optimization (avoiding unnecessary allocations and copies), idiomatic refactoring (use of `Result`/`Option`), and security auditing (safe use of `unsafe` and concurrency). The goal is to transform the AI from a generic coding assistant into a specialized co-pilot that understands and respects Rust's safety and performance philosophy.

## Examples
```
**1. Performance Optimization (Critical Path Analysis)**
**Role:** Act as a Senior Rust Performance Engineer.
**Task:** Analyze the Rust code block provided below. Identify the "critical path" (hot path) and suggest optimizations that reduce memory allocation, minimize unnecessary copies, and improve iterator efficiency. Optimize for latency, not just throughput.
**Code:** [INSERT RUST CODE HERE]
**Constraint:** Maintain memory safety and code readability.

**2. Idiomatic Refactoring**
**Role:** Act as an experienced Rust Code Reviewer.
**Task:** Refactor the following Rust code to make it more idiomatic, using Rust design patterns (such as `Result`, `Option`, `match`), and following the `clippy` guidelines. Explain each change and the Rust principle it reinforces.
**Code:** [INSERT RUST CODE HERE]
**Constraint:** The refactored code must pass `cargo clippy -- -D warnings`.

**3. Security Audit**
**Role:** Act as a Software Security Auditor specialized in Rust.
**Task:** Review the provided Rust module for security vulnerabilities, such as errors in the use of `unsafe`, concurrency issues (data races), or flaws in input/output (I/O) handling. Suggest fixes and explain the security flaw that each fix mitigates.
**Code:** [INSERT RUST CODE HERE]
**Constraint:** Prioritize eliminating any use of `unsafe` that is not strictly necessary.

**4. Unit Test Generation**
**Role:** Act as a Software Test Engineer.
**Task:** Generate a comprehensive set of unit tests for the provided Rust function. Include test cases for valid inputs, edge cases, and error handling (`panic` or `Result::Err`).
**Function:** [INSERT RUST FUNCTION HERE]
**Constraint:** Use the `#[cfg(test)]` macro and Rust's standard test module.

**5. Compiler Error Explanation**
**Role:** Act as a Rust Language Tutor.
**Task:** Explain the Rust compiler error provided below. Describe the root cause of the error (for example, a violated *ownership* rule, a *lifetime* error), and provide a minimal, functional code solution.
**Error Message:** [INSERT COMPILER ERROR MESSAGE HERE]
**Constraint:** The explanation must be clear and didactic, focusing on the core Rust concept.

**6. Project Structure Generation**
**Role:** Act as a Software Architect.
**Task:** Generate the file and directory structure (`Cargo.toml`, `src/main.rs`, `src/lib.rs`, modules, etc.) for a new Rust project that will implement an asynchronous web server using `tokio` and `actix-web`.
**Project Goal:** RESTful server for user management.
**Constraint:** The project must follow Rust's module conventions and be ready to be compiled with `cargo build`.

**7. Documentation and Example**
**Role:** Act as a Technical Writer.
**Task:** Write complete documentation for the provided Rust function, including a usage example in `doctest` format. The documentation must clearly explain the parameters, the return value, and the function's behavior.
**Function:** [INSERT RUST FUNCTION HERE]
**Constraint:** Use Rust's standard documentation syntax (`///`).
```

## Best Practices
**Be Specific with the Role and Context (Role and Context Specificity):** Always define a specialized role for the AI (e.g., "Senior Rust Performance Engineer," "Security Auditor"). Provide the complete code block or the compiler error message to ensure the AI has the necessary context for the *ownership* and *lifetimes* rules. **Focus on Idiomatic Concepts:** Direct the AI to use Rust design patterns, such as `Result`, `Option`, `match`, and to follow the `clippy` guidelines. Ask for explanations of how the suggestions adhere to Rust's philosophy. **Cross-Validate with Native Tools:** Use the prompts to generate code or suggestions, but always validate the result with Rust's native tools, such as `cargo check`, `cargo clippy`, and unit tests (`cargo test`).

## Use Cases
**High-Performance Code Optimization:** Identify and fix performance bottlenecks in Rust code, especially in loops, iterations, and collection handling, focusing on avoiding *heap* allocations. **Refactoring for Idiomaticity:** Convert functional but non-idiomatic code into code that follows the best practices and patterns of the Rust ecosystem, improving readability and maintainability. **Automated Security Auditing:** Use the AI to review critical modules, especially those that use `unsafe` or handle concurrency, in search of *data races* or other security vulnerabilities. **Comprehensive Test Generation:** Create robust unit and integration tests that cover normal use cases, edge cases, and error handling, ensuring software quality. **Accelerating the Learning Curve:** Use the AI as a specialized tutor to explain complex compiler errors related to *ownership*, *borrowing*, and *lifetimes*, accelerating mastery of the language.

## Pitfalls
**Generic Prompts:** Using prompts like "Optimize this code" without specifying what to optimize (memory, CPU, I/O) or without mentioning Rust's *ownership* rules. **Ignoring Compiler Output:** Not providing the complete Rust compiler error message (which is highly informative) to the AI, resulting in incorrect or incomplete fixes. **Lack of *Lifetime* Context:** Asking the AI to refactor code involving complex *lifetimes* without providing the full context of the surrounding functions and structures. **Blindly Trusting `unsafe`:** Allowing the AI to suggest the use of `unsafe` blocks without a clear justification and an explanation of how memory safety is maintained, violating Rust's core philosophy.

## URL
[https://github.com/Ranrar/rustic-prompt](https://github.com/Ranrar/rustic-prompt)
