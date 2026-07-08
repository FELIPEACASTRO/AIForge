# C++ Programming Prompts

## Description
**C++ Programming Prompts** refer to structured and detailed instructions provided to Large Language Models (LLMs) to generate, analyze, refactor, document, or debug C++ code. Due to the complexity, strict syntax, and focus on performance and memory management of C++, prompt engineering for this language requires a high degree of precision and context. The effective technique is based on four main pillars: **Persona** (defining the AI's role), **Context** (providing technical details such as the C++ standard version, libraries, and system constraints), **Task** (the clear objective), and **Format** (the desired output structure). Recent research (2023-2025) emphasizes the **synergistic** approach to prompts, combining techniques such as *Chain-of-Thought* (CoT) and *Few-Shot* to improve the precision and safety of the generated code, especially in critical domains such as embedded systems and game development. The goal is to mitigate the tendency of LLMs to produce functional but insecure or inefficient code, which is a particular risk in C++ development.

## Examples
```
**1. Generation of Optimized Code for an Embedded System**
```
Persona: You are a senior C++ software engineer, an expert in embedded systems with memory constraints.
Context: I am using C++17 on an ARM Cortex-M4 microcontroller. The function must be optimized for speed and have a minimal memory footprint.
Task: Write a C++ function that implements a Moving Average Filter for an array of 100 16-bit integers. Use `std::array` and avoid dynamic memory allocation.
Format: Provide only the complete C++ source code, including the function and a small example `main`.
```

**2. Refactoring and Modernization of Legacy Code**
```
Persona: You are a C++ software architect focused on code modernization.
Context: The following C++ code snippet uses raw pointers and manual allocation. [Include legacy code snippet]
Task: Refactor the code to use modern C++ (C++20) smart pointers, such as `std::unique_ptr` and `std::shared_ptr`, to ensure safe memory management. Keep the original functionality.
Format: Present the refactored code and a brief explanation of the changes in a numbered list.
```

**3. Security Analysis and Code Review**
```
Persona: You are a software security expert, focused on C/C++ vulnerabilities.
Context: Analyze the following C++ code that handles user input. [Include C++ code]
Task: Identify potential security vulnerabilities, such as buffer overflows, race conditions, or memory leaks. Suggest specific fixes and explain the risk of each vulnerability.
Format: Use a Markdown table with the columns: 'Vulnerability', 'Severity (High/Medium/Low)', 'Explanation', 'Suggested Fix'.
```

**4. Generation of Unit Tests with a Specific Framework**
```
Persona: You are a QA engineer with experience in C++ unit testing.
Context: The C++ class `Calculator` has methods for addition, subtraction, and division. [Include the class header `Calculator.h`]
Task: Generate a comprehensive set of unit tests for the `Calculator` class using the **Google Test** framework. Include edge case tests, such as division by zero.
Format: Provide the complete test `.cpp` file, ready for compilation.
```

**5. Explanation of Advanced C++ Concepts**
```
Persona: You are a didactic C++ instructor focused on clarity and practical examples.
Context: My level is intermediate and I am learning C++20.
Task: Explain the concept of **Coroutines** in C++20. Include a small code example that demonstrates the use of a simple coroutine for an asynchronous operation.
Format: Use clear paragraphs for the explanation and a well-commented C++ code block for the example.
```

**6. Code Generation with Design Pattern Constraints**
```
Persona: You are an experienced C++ developer skilled in Design Patterns.
Context: I am implementing a logging system.
Task: Write the C++ implementation (.h and .cpp files) of a thread-safe **Singleton Design Pattern** for a Log class. Use the *Meyers' Singleton* (Magic Static) initialization to ensure thread safety and lazy initialization.
Format: Provide the two files (`Log.h` and `Log.cpp`) separately.
```
```

## Best Practices
**Define the Persona (Role-Playing):** Always begin the prompt by defining the AI's role (e.g., "You are a senior C++ software engineer, an expert in embedded systems and performance optimization"). This aligns the response to the desired level of knowledge and style.
**Provide Detailed Context:** Include as much context as possible, such as the C++ version (C++17, C++20, C++23), the libraries to be used (Boost, Qt, STL, etc.), the target environment (Linux, Windows, microcontroller), and performance or memory constraints.
**Specify the Output Format:** Explicitly request the desired format (e.g., "Provide only the code, without explanations", "Use Doxygen comments for documentation", "Present the analysis in a Markdown table").
**Break Down Complex Tasks:** Instead of asking to "Create a C++ web server", break it down into steps: "1. Write the TCP socket class", "2. Implement the event loop", "3. Add error handling".
**Include Examples (Few-Shot):** To ensure the generated code follows a specific style or pattern from your codebase, include a small snippet of existing C++ code as an example.
**Focus on Security and Robustness:** Explicitly request secure and robust code, asking for checks against *memory leaks*, *buffer overflows*, and the use of modern practices (e.g., `std::unique_ptr`, `std::span`).

## Use Cases
**Optimized Code Generation:** Create C++ functions and classes that meet strict performance and resource usage requirements, common in *High-Frequency Trading* or *Game Engines*.
**Legacy Code Modernization:** Refactor old codebases (C++98/03) to modern standards (C++17/20/23), replacing raw pointers with *smart pointers* and using features such as *Concepts* and *Modules*.
**Code Analysis for Security:** Identify and fix C/C++-specific security vulnerabilities, such as *buffer overflows* and *integer overflows*, before human review.
**Unit Test Generation:** Quickly create comprehensive test suites using frameworks such as Google Test or Catch2, including edge cases and stress tests.
**Technical Documentation:** Generate Doxygen documentation or clear, consistent code comments for large C++ projects, ensuring long-term maintainability.
**Explanation of Advanced Concepts:** Use the AI as a tutor to explain complex C++ features (e.g., *template metaprogramming*, *variadic templates*, *Coroutines*) with practical, working examples.

## Pitfalls
**Vague or Generic Prompts:** Asking "Write some C++ code" without specifying the standard version (C++11 vs C++20), the environment, or the libraries. This results in generic, potentially obsolete, or incompatible code.
**Ignoring Memory Management:** Not mentioning smart pointers or memory allocation. The LLM may generate code with raw `new` and `delete`, introducing *memory leaks* or segmentation faults.
**Lack of Performance Context:** Not specifying the need for optimization. The LLM may use high-level data structures or algorithms that are slow or consume a lot of memory, which is critical in C++.
**Blindly Trusting Security:** The generated code may appear correct but contain security vulnerabilities (e.g., insecure use of `strcpy` or `scanf`). It is crucial to explicitly request a security check.
**Task Overload:** Trying to refactor, document, and optimize a large block of code in a single prompt. This confuses the model and reduces the quality of the output. Break it into smaller, sequential prompts.

## URL
[https://blogs.sw.siemens.com/thought-leadership/prompt-engineering-part-2-best-practices-for-software-developers-in-digital-industries/](https://blogs.sw.siemens.com/thought-leadership/prompt-engineering-part-2-best-practices-for-software-developers-in-digital-industries/)
