# Python Programming Prompts

## Description
**Python Programming Prompts** are structured, detailed instructions provided to a Large Language Model (LLM) with the goal of generating, debugging, refactoring, documenting, or explaining code in the Python language. The effectiveness of these prompts depends on clarity, the context provided, and the application of prompt engineering techniques specific to coding tasks. They transform the LLM from a generic text generator into a highly specialized programming assistant, capable of handling tasks ranging from creating small scripts to architecting complex systems, always with a focus on adherence to code standards such as PEP 8. Recent research (2023-2025) emphasizes the importance of iterative prompts, the inclusion of tests, and the clear definition of environment and performance constraints.

## Examples
```
**1. Function Generation with Unit Test:**
```
**Role:** You are a Python software engineer.
**Task:** Write a Python function `calculate_median(data_list)` that takes a list of numbers and returns the median.
**Constraints:** The function must handle both even and odd-sized lists. Include a unit test using the `unittest` module that asserts the median of `[1, 2, 3, 4, 5]` is `3` and the median of `[1, 2, 3, 4]` is `2.5`.
**Output Format:** Only the Python code block.
```

**2. Refactoring for Optimization:**
```
**Task:** Refactor the following Python code to improve its performance and adhere to PEP 8. The goal is to replace the nested loops with a more Pythonic and efficient approach, preferably using a dictionary or set for O(1) lookups.
**Code to Refactor:**
```python
def find_duplicates(list1, list2):
    duplicates = []
    for item1 in list1:
        for item2 in list2:
            if item1 == item2 and item1 not in duplicates:
                duplicates.append(item1)
    return duplicates
```
**Output Format:** Only the refactored Python code block.
```

**3. Data Analysis Script Generation:**
```
**Role:** You are a Data Scientist.
**Task:** Write a Python script using the Pandas library to perform the following steps:
1. Load the CSV file named 'sales_data.csv' into a DataFrame.
2. Calculate the mean of the 'Revenue' column, grouped by the 'Region' column.
3. Print the resulting Series.
**Constraints:** Assume the CSV file exists in the current directory. Do not use any external functions.
**Output Format:** Only the complete Python script.
```

**4. Error Debugging:**
```
**Task:** The following Python code is raising a `KeyError: 'city'`. Analyze the code and the traceback, identify the bug, and provide the corrected code.
**Code:**
```python
data = [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25, 'city': 'New York'}]
for item in data:
    print(item['city'])
```
**Traceback:**
```
Traceback (most recent call last):
  File "script.py", line 3, in <module>
    print(item['city'])
KeyError: 'city'
```
**Output Format:** First, a brief explanation of the bug, then the corrected Python code block using a `try-except` block or `.get()`.
```

**5. Complex Code Explanation:**
```
**Task:** Explain the following Python code snippet line by line, focusing on the use of the `__call__` method and the concept of a callable class instance.
**Code:**
```python
class Multiplier:
    def __init__(self, factor):
        self.factor = factor
    def __call__(self, number):
        return number * self.factor
m = Multiplier(5)
result = m(10)
```
**Output Format:** A detailed, didactic explanation in Portuguese, formatted as a numbered list.
```

**6. Documentation Generation:**
```
**Task:** Generate a comprehensive docstring in the Google Python Style Guide format for the following function. The docstring must include a description, arguments, return value, and an example of usage.
**Function:**
```python
def connect_to_database(host, port=5432, timeout=5):
    """Connects to the PostgreSQL database."""
    # implementation details...
    return connection_object
```
**Output Format:** Only the docstring content.
```
```

## Best Practices
**1. Be Specific and Structured:**
   - **Define the Role:** Start the prompt by defining the LLM's role (e.g., "You are a senior Python software engineer...").
   - **Specify the Version:** Include the Python version and libraries (e.g., "Use Python 3.11 and the Pandas library").
   - **Output Format:** Ask for the code inside Markdown code blocks (` ```python `) and instruct the model not to include unnecessary explanations unless requested.

**2. Provide Context and Constraints:**
   - **Data Schema:** If applicable, provide the data schema, variable names, and existing classes or functions.
   - **Constraints:** Include performance, security (e.g., "The code must be optimized for O(n)"), or style constraints (e.g., "Follow PEP 8").
   - **Tests:** Ask the model to include unit tests (`unittest` or `pytest`) for the generated code.

**3. Use Advanced Techniques:**
   - **Chain-of-Thought (CoT):** For complex tasks, ask the model to "think out loud" or describe the implementation plan before generating the code.
   - **Few-Shot Learning:** Provide one or two examples of problem/solution pairs to guide the style and complexity of the code.
   - **Iteration and Refinement:** Instead of a single long prompt, use short, iterative prompts to refine the code (e.g., "Refactor the `process_data` function to use a `list comprehension`").

## Use Cases
**1. Rapid Prototyping:**
   - Quickly create functions, classes, or utility scripts to test an idea or concept, reducing initial development time.

**2. Debugging and Error Correction:**
   - Insert a code snippet with an error and the traceback message so that the LLM identifies the cause and suggests a fix.

**3. Code Refactoring and Optimization:**
   - Request improvements to existing code for greater efficiency, readability (adherence to PEP 8), or modernization (e.g., converting loops into `list comprehensions`).

**4. Language Translation:**
   - Convert code from another language (e.g., JavaScript, R) to Python, preserving the logic and functionality.

**5. Documentation and Test Generation:**
   - Automatically create docstrings (in Sphinx, NumPy, or Google format) and unit tests (using `unittest` or `pytest`) for existing functions and modules.

**6. Explanation and Learning:**
   - Ask the LLM to explain how a complex code snippet works, a specific Python concept (e.g., *decorators*, *generators*), or the purpose of a library.

## Pitfalls
**1. Lack of Context:**
   - **Mistake:** Providing vague prompts (e.g., "Write Python code to process data") without specifying the input format, the expected result, or the libraries to be used.
   - **Consequence:** Generation of generic, inefficient code, or code that does not integrate with the existing project.

**2. Over-reliance and No Verification:**
   - **Mistake:** Assuming that LLM-generated code is always correct and functional, especially for complex logic or security.
   - **Consequence:** Introduction of bugs, security vulnerabilities, or unoptimized code into the project. **Always verify and test the generated code.**

**3. Ignoring Token Limits:**
   - **Mistake:** Providing very large codebases for debugging or refactoring in a single prompt, exceeding the model's context limit.
   - **Consequence:** The model ignores parts of the code or generates an incomplete response. **Solution:** Break the task into smaller, iterative prompts.

**4. Not Specifying Style:**
   - **Mistake:** Not mentioning coding standards (e.g., PEP 8) or the project's naming conventions.
   - **Consequence:** Functional code that is inconsistent with the rest of the project, requiring subsequent manual refactoring.

**5. "Black Box" Prompts:**
   - **Mistake:** Asking only for the final result without requesting the reasoning process (Chain-of-Thought).
   - **Consequence:** Difficulty debugging or understanding the logic behind a complex solution, missing the learning opportunity.

## URL
[https://github.com/potpie-ai/potpie/wiki/How-to-write-good-prompts-for-generating-code-from-LLMs](https://github.com/potpie-ai/potpie/wiki/How-to-write-good-prompts-for-generating-code-from-LLMs)
